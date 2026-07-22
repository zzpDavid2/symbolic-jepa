"""
Evaluation utilities for symbolic regression.

Includes constant fitting (BFGS), R² scoring, token accuracy,
and algebraic equivalence checking.
"""

import signal
import threading

import numpy as np
import sympy as sp
from scipy.optimize import minimize
from tqdm.auto import tqdm

from symbolic_jepa.tokenizer import prefix_to_sympy


def cleanup_eval_pool():
    """No-op — kept for backward compatibility with notebook code.

    Signal-based timeouts don't use a pool, so nothing to clean up.
    """
    pass


def _run_with_timeout(fn, timeout):
    """Run fn() with a hard timeout. Returns None on timeout/error.

    On Unix main thread: uses SIGALRM to actually interrupt stuck work
    (sympy simplify, BFGS fitting, etc.). No zombie threads.
    Fallback: daemon thread (stuck work may linger briefly).
    """
    if (hasattr(signal, 'SIGALRM')
            and threading.current_thread() is threading.main_thread()):
        return _signal_timeout(fn, timeout)
    return _thread_timeout(fn, timeout)


def _signal_timeout(fn, timeout):
    """SIGALRM-based timeout — actually kills stuck work."""
    def _handler(signum, frame):
        raise TimeoutError()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        return fn()
    except (TimeoutError, Exception):
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _thread_timeout(fn, timeout):
    """Thread-based timeout fallback (non-Unix / non-main-thread)."""
    result = [None]
    ok = [False]

    def _target():
        try:
            result[0] = fn()
            ok[0] = True
        except Exception:
            pass

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0] if ok[0] else None


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
    # logits[:, 1:-1, :] → predictions from <sos> onward (skipping data-token)
    # targets[:, 1:]     → ground-truth after <sos>
    pred = logits[:, 1:-1, :].argmax(dim=-1)   # (batch, seq-1)
    tgt = targets[:, 1:]                        # (batch, seq-1)
    mask = (tgt != pad_id)
    correct = ((pred == tgt) & mask).float().sum()
    total = mask.float().sum()
    return (correct / (total + 1e-10)).item()


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

    Handles commutativity (SymPy normalizes) and constant permutation
    (c_0, c_1, ... may map differently between pred and gt).
    """
    try:
        pred_expr, pred_consts = prefix_to_sympy(pred_str)
        gt_expr, gt_consts = prefix_to_sympy(gt_str)
    except Exception:
        return False

    def _check():
        # Fast path: identical SymPy expressions (handles commutativity)
        if pred_expr.equals(gt_expr):
            return True

        # Try simplify on the difference (works when constants align)
        diff = sp.simplify(pred_expr - gt_expr)
        if diff.is_zero is True:
            return True

        # Constant permutation: if same number of constants, try all
        # permutations of pred constants to match gt constants.
        # (Constants are fittable, so c_0*x + c_1 ≡ c_1*x + c_0.)
        n_pred, n_gt = len(pred_consts), len(gt_consts)
        if n_pred == n_gt and 0 < n_pred <= 6:
            from itertools import permutations
            for perm in permutations(pred_consts):
                sub = dict(zip(perm, gt_consts))
                remapped = pred_expr.subs(sub)
                if remapped.equals(gt_expr):
                    return True
                diff2 = sp.simplify(remapped - gt_expr)
                if diff2.is_zero is True:
                    return True

        return False

    result = _run_with_timeout(_check, timeout)
    return result is True


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: list[tuple[str, str]],
    dataset,
    tokenizer,
    n_fit_points: int = 200,
    fit_timeout: int = 2,
) -> dict:
    """Evaluate a list of (gt_prefix, pred_prefix) pairs.

    Args:
        predictions: List of (ground_truth_prefix, predicted_prefix) tuples.
        dataset: PointCloudDataset (for sampling evaluation points).
        tokenizer: PrefixTokenizer.
        n_fit_points: Number of points for constant fitting.
        fit_timeout: Timeout in seconds for each R² fit attempt.

    Returns:
        Dict with 'exact_match', 'token_accuracy', 'algebraic_equiv',
        'mean_r2', 'r2_above_0.9', and per-sample 'details'.
    """
    exact_matches = []
    token_accs = []
    algebraic_matches = []
    r2_scores = []
    details = []

    pbar = tqdm(predictions, desc='evaluate', leave=False)
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

        # Algebraic equivalence — skip expensive simplify for exact matches
        if exact:
            equiv = 1
        else:
            equiv = int(equations_equivalent(pred_str, gt_str))
        algebraic_matches.append(equiv)

        # Parse check
        parseable = False
        try:
            prefix_to_sympy(pred_str)
            parseable = True
        except Exception:
            pass

        # R² via constant fitting (thread-based timeout)
        r2 = None
        if parseable and i < len(dataset.samples):
            expr_obj = dataset.samples[i]['expr']

            def _fit_r2():
                pred_expr, constants = prefix_to_sympy(pred_str)
                cloud = expr_obj.sample(n_fit_points)
                finite_mask = np.isfinite(cloud).all(axis=1)
                cloud = cloud[finite_mask]
                if len(cloud) < 50:
                    return None
                n_vars = len(expr_obj.variables)
                X = cloud[:, :n_vars]
                Y = cloud[:, n_vars]
                var_syms = [sp.Symbol(f'x{j+1}') for j in range(n_vars)]
                _, _, r2_val = fit_constants(pred_expr, constants, X, Y, var_syms)
                return r2_val

            r2 = _run_with_timeout(_fit_r2, fit_timeout)

        if r2 is not None and np.isfinite(r2):
            r2_scores.append(r2)

        details.append({
            'gt': gt_str, 'pred': pred_str,
            'exact': exact, 'equiv': equiv, 'r2': r2, 'parseable': parseable,
        })

        # Update progress bar with running stats
        if (i + 1) % 20 == 0:
            em = np.mean(exact_matches) * 100
            eq = np.mean(algebraic_matches) * 100
            pbar.set_postfix(exact=f'{em:.0f}%', equiv=f'{eq:.0f}%')

    n = len(predictions)
    results = {
        'exact_match': np.mean(exact_matches) if exact_matches else 0,
        'token_accuracy': np.mean(token_accs) if token_accs else 0,
        'algebraic_equiv': np.mean(algebraic_matches) if algebraic_matches else 0,
        'mean_r2': np.mean(r2_scores) if r2_scores else float('nan'),
        'median_r2': np.median(r2_scores) if r2_scores else float('nan'),
        'r2_above_0.9': np.mean([r > 0.9 for r in r2_scores]) if r2_scores else 0,
        'n_parseable': sum(1 for d in details if d['parseable']),
        'n_total': n,
        'details': details,
    }
    return results
