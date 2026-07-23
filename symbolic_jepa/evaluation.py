"""
Evaluation utilities for symbolic regression.

Includes constant fitting (BFGS), R² scoring, token accuracy,
and algebraic equivalence checking.
"""

import numpy as np
import sympy as sp
from scipy.optimize import minimize
from tqdm.auto import tqdm

from symbolic_jepa.tokenizer import prefix_to_sympy


def cleanup_eval_pool():
    """No-op — kept for backward compatibility with notebook code."""
    pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_score(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = float(np.sum((Y - Y_pred) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    return 1 - ss_res / (ss_tot + 1e-10)


def teacher_forced_accuracy(logits, targets, pad_id: int) -> float:
    """Fraction of non-pad positions where argmax matches target.

    Excludes the trivial data-token → <sos> prediction (position 0).

    Args:
        logits: (batch, 1+seq, vocab) — includes data-token position.
        targets: (batch, seq) — ground-truth token IDs.
        pad_id: Token ID used for padding.
    """
    correct, total = teacher_forced_counts(logits, targets, pad_id)
    return correct / (total + 1e-10)


def teacher_forced_counts(logits, targets, pad_id: int) -> tuple[float, float]:
    """Return (n_correct, n_total) for token-level accuracy.

    Same logic as teacher_forced_accuracy but returns raw counts
    for proper aggregation across batches of different lengths.
    """
    pred = logits[:, 1:-1, :].argmax(dim=-1)   # (batch, seq-1)
    tgt = targets[:, 1:]                        # (batch, seq-1)
    mask = (tgt != pad_id)
    correct = ((pred == tgt) & mask).float().sum().item()
    total = mask.float().sum().item()
    return correct, total


# ---------------------------------------------------------------------------
# Constant fitting
# ---------------------------------------------------------------------------

def fit_constants(expr, constants, X, Y, var_syms, maxiter=100):
    """Fit fittable constants in a predicted expression using L-BFGS-B.

    Args:
        expr: SymPy expression (may contain c_0, c_1, ... symbols).
        constants: List of SymPy Symbol objects for fittable constants.
        X: (n_points, n_vars) input data.
        Y: (n_points,) target output.
        var_syms: List of SymPy Symbols for input variables.
        maxiter: Maximum BFGS iterations.

    Returns:
        (fitted_dict, Y_pred, r2) or (None, None, -inf) on failure.
    """
    if len(constants) == 0:
        f = sp.lambdify(var_syms, expr, 'numpy')
        try:
            Y_pred = np.broadcast_to(
                np.asarray(f(*X.T), dtype=float), Y.shape
            ).copy()
            return {}, Y_pred, r2_score(Y, Y_pred)
        except Exception:
            return None, None, -np.inf

    f = sp.lambdify(list(var_syms) + list(constants), expr, 'numpy')

    def loss(c):
        with np.errstate(all='ignore'):
            try:
                p = np.asarray(f(*X.T, *c), dtype=float)
                return float(np.mean((p - Y) ** 2)) if np.all(np.isfinite(p)) else 1e10
            except Exception:
                return 1e10

    r = minimize(loss, np.ones(len(constants)), method='L-BFGS-B',
                 options={'maxiter': maxiter})

    if not np.isfinite(r.fun) or r.fun >= 1e9:
        return None, None, -np.inf

    fitted = dict(zip([str(c) for c in constants], r.x))
    with np.errstate(all='ignore'):
        Y_pred = np.asarray(f(*X.T, *r.x), dtype=float)
    if not np.all(np.isfinite(Y_pred)):
        return None, None, -np.inf
    return fitted, Y_pred, r2_score(Y, Y_pred)


# ---------------------------------------------------------------------------
# Equivalence checking
# ---------------------------------------------------------------------------

def equations_equivalent(pred_str: str, gt_str: str, timeout: int = 2) -> bool:
    """Check if two prefix strings are algebraically equivalent.

    Uses only fast checks (SymPy .equals and expand, no sp.simplify)
    to avoid hanging on pathological expressions.  Handles commutativity
    (SymPy normalizes) and constant permutation.
    """
    try:
        pred_expr, pred_consts = prefix_to_sympy(pred_str)
        gt_expr, gt_consts = prefix_to_sympy(gt_str)
    except Exception:
        return False

    # Fast path: identical SymPy expressions (handles commutativity)
    try:
        if pred_expr.equals(gt_expr):
            return True
    except Exception:
        pass

    # Try expand on the difference (fast, catches most algebraic identities)
    try:
        diff = sp.expand(pred_expr - gt_expr)
        if diff is sp.S.Zero or diff == 0:
            return True
    except Exception:
        pass

    # Constant permutation: if same number of constants, try all
    # permutations of pred constants to match gt constants.
    n_pred, n_gt = len(pred_consts), len(gt_consts)
    if n_pred == n_gt and 0 < n_pred <= 4:
        from itertools import permutations
        for perm in permutations(pred_consts):
            sub = dict(zip(perm, gt_consts))
            remapped = pred_expr.subs(sub)
            try:
                if remapped.equals(gt_expr):
                    return True
                diff2 = sp.expand(remapped - gt_expr)
                if diff2 is sp.S.Zero or diff2 == 0:
                    return True
            except Exception:
                continue

    return False


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

# Deterministic seeds for R² evaluation, independent of model seed.
# Two separate seeds so fit and eval clouds are independent.
_R2_FIT_SEED  = 777_777_773
_R2_EVAL_SEED = 555_555_557


def evaluate_predictions(
    predictions: list[tuple[str, str]],
    dataset,
    tokenizer,
    n_fit_points: int = 200,
    n_eval_points: int = 200,
    r2_equiv_thresh: float = 0.999,
) -> dict:
    """Evaluate a list of (gt_prefix, pred_prefix) pairs.

    Equivalence is determined by held-out R² after constant fitting:
    constants are fitted on one deterministic point cloud, then R² is
    computed on a separate held-out cloud.  Both are seeded per-sample
    for reproducibility.

    Args:
        predictions: List of (ground_truth_prefix, predicted_prefix) tuples.
        dataset: PointCloudDataset (for sampling evaluation points).
        tokenizer: PrefixTokenizer.
        n_fit_points: Number of points for constant fitting.
        n_eval_points: Number of held-out points for R² evaluation.
        r2_equiv_thresh: R² threshold for declaring equivalence (default 0.999).

    Returns:
        Dict with 'exact_match', 'token_accuracy', 'algebraic_equiv',
        'mean_r2', 'r2_above_0.9', and per-sample 'details'.
    """
    exact_matches = []
    token_accs = []
    equiv_matches = []
    r2_all = []  # one entry per prediction (None if not computable)
    details = []

    pbar = tqdm(predictions, desc='evaluate', leave=True)
    for i, (gt_str, pred_str) in enumerate(pbar):
        # Exact match
        exact = int(pred_str.strip() == gt_str.strip())
        exact_matches.append(exact)

        # Token accuracy
        pred_tokens = pred_str.split()
        gt_tokens = gt_str.split()
        min_len = min(len(pred_tokens), len(gt_tokens))
        if min_len > 0:
            hits = sum(p == g for p, g in zip(pred_tokens[:min_len], gt_tokens[:min_len]))
            token_accs.append(hits / max(len(pred_tokens), len(gt_tokens)))

        # Parse both expressions
        parseable = False
        pred_expr = None
        pred_consts = None
        gt_expr = None
        try:
            pred_expr, pred_consts = prefix_to_sympy(pred_str)
            parseable = True
        except Exception:
            pass

        try:
            gt_expr, _ = prefix_to_sympy(gt_str)
        except Exception:
            pass

        # ── Equivalence cascade: fast sympy first, then R² ──
        equiv = exact  # start with exact match

        # Level 1: sp.expand on difference (milliseconds, no risk of hanging)
        if not equiv and pred_expr is not None and gt_expr is not None:
            try:
                diff = sp.expand(pred_expr - gt_expr)
                if diff is sp.S.Zero or diff == 0:
                    equiv = 1
            except Exception:
                pass

        # Level 2: R² via constant fitting on fit cloud, evaluated on held-out cloud.
        # Uses deterministic RNGs seeded per-sample for reproducibility.
        r2 = None
        if parseable and i < len(dataset.samples):
            try:
                expr_obj = dataset.samples[i]['expr']
                n_vars = len(expr_obj.variables)
                var_syms = [sp.Symbol(f'x{j+1}') for j in range(n_vars)]

                # Fit cloud: fit constants
                fit_rng = np.random.RandomState(_R2_FIT_SEED + i)
                fit_cloud = expr_obj.sample(n_fit_points, rng=fit_rng)
                fit_mask = np.isfinite(fit_cloud).all(axis=1)
                fit_cloud = fit_cloud[fit_mask]

                if len(fit_cloud) >= 50:
                    X_fit = fit_cloud[:, :n_vars]
                    Y_fit = fit_cloud[:, n_vars]
                    fitted_dict, _, _ = fit_constants(
                        pred_expr, pred_consts, X_fit, Y_fit, var_syms,
                    )

                    if fitted_dict is not None:
                        # Substitute fitted constants into expression
                        subs = {c: fitted_dict[str(c)] for c in pred_consts
                                if str(c) in fitted_dict}
                        fitted_expr = pred_expr.subs(subs) if subs else pred_expr
                        f_eval = sp.lambdify(var_syms, fitted_expr, 'numpy')

                        # Eval cloud: compute held-out R²
                        eval_rng = np.random.RandomState(_R2_EVAL_SEED + i)
                        eval_cloud = expr_obj.sample(n_eval_points, rng=eval_rng)
                        eval_mask = np.isfinite(eval_cloud).all(axis=1)
                        eval_cloud = eval_cloud[eval_mask]

                        if len(eval_cloud) >= 50:
                            X_eval = eval_cloud[:, :n_vars]
                            Y_eval = eval_cloud[:, n_vars]
                            with np.errstate(all='ignore'):
                                Y_pred = np.broadcast_to(
                                    np.asarray(f_eval(*X_eval.T), dtype=float),
                                    Y_eval.shape,
                                ).copy()
                            if np.all(np.isfinite(Y_pred)):
                                r2 = r2_score(Y_eval, Y_pred)
            except Exception:
                r2 = None

        r2_all.append(r2)

        if r2 is not None and np.isfinite(r2):
            if not equiv and r2 >= r2_equiv_thresh:
                equiv = 1

        equiv_matches.append(equiv)

        details.append({
            'gt': gt_str, 'pred': pred_str,
            'exact': exact, 'equiv': equiv, 'r2': r2, 'parseable': parseable,
        })

        # Update progress bar with running stats
        if (i + 1) % 10 == 0 or i == len(predictions) - 1:
            em = np.mean(exact_matches) * 100
            eq = np.mean(equiv_matches) * 100
            pbar.set_postfix(exact=f'{em:.0f}%', equiv=f'{eq:.0f}%')

    n = len(predictions)
    finite_r2 = [r for r in r2_all if r is not None and np.isfinite(r)]
    results = {
        'exact_match': np.mean(exact_matches) if exact_matches else 0,
        'token_accuracy': np.mean(token_accs) if token_accs else 0,
        'algebraic_equiv': np.mean(equiv_matches) if equiv_matches else 0,
        'mean_r2': np.mean(finite_r2) if finite_r2 else float('nan'),
        'median_r2': np.median(finite_r2) if finite_r2 else float('nan'),
        # R²>0.9 out of ALL predictions (not just parseable)
        'r2_above_0.9': sum(1 for r in r2_all if r is not None and r > 0.9) / max(n, 1),
        'n_parseable': sum(1 for d in details if d['parseable']),
        'n_total': n,
        'details': details,
    }
    return results
