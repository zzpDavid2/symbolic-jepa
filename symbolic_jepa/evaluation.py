"""
Evaluation utilities for symbolic regression.

Includes constant fitting (BFGS), R² scoring, token accuracy,
and algebraic equivalence checking.
"""

import signal
import numpy as np
import sympy as sp
from scipy.optimize import minimize

from symbolic_jepa.tokenizer import prefix_to_sympy


class _Timeout:
    """Context manager that raises TimeoutError after `seconds`."""
    def __init__(self, seconds):
        self.seconds = seconds
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
    def __exit__(self, *args):
        signal.alarm(0)
    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError


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

    Args:
        logits: (batch, 1+seq, vocab) — includes data-token position.
        targets: (batch, seq) — ground-truth token IDs.
        pad_id: Token ID used for padding.
    """
    pred = logits[:, :-1, :].argmax(dim=-1)  # (batch, seq)
    mask = (targets != pad_id)
    correct = ((pred == targets) & mask).float().sum()
    total = mask.float().sum()
    return (correct / (total + 1e-10)).item()


# ---------------------------------------------------------------------------
# Constant fitting
# ---------------------------------------------------------------------------

def fit_constants(expr, constants, X, Y, var_syms):
    """Fit fittable constants in a predicted expression using L-BFGS-B.

    Args:
        expr: SymPy expression (may contain c_0, c_1, ... symbols).
        constants: List of SymPy Symbol objects for fittable constants.
        X: (n_points, n_vars) input data.
        Y: (n_points,) target output.
        var_syms: List of SymPy Symbols for input variables.

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
                 options={'maxiter': 300})

    if not np.isfinite(r.fun) or r.fun >= 1e9:
        return None, None, -np.inf

    fitted = dict(zip([str(c) for c in constants], r.x))
    Y_pred = np.asarray(f(*X.T, *r.x), dtype=float)
    return fitted, Y_pred, r2_score(Y, Y_pred)


# ---------------------------------------------------------------------------
# Equivalence checking
# ---------------------------------------------------------------------------

def equations_equivalent(pred_str: str, gt_str: str, timeout: int = 5) -> bool:
    """Check if two prefix strings are algebraically equivalent."""
    try:
        pred_expr, _ = prefix_to_sympy(pred_str)
        gt_expr, _ = prefix_to_sympy(gt_str)
    except Exception:
        return False

    try:
        with _Timeout(timeout):
            diff = sp.simplify(pred_expr - gt_expr)
        return diff.is_zero is True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: list[tuple[str, str]],
    dataset,
    tokenizer,
    n_fit_points: int = 1000,
) -> dict:
    """Evaluate a list of (gt_prefix, pred_prefix) pairs.

    Args:
        predictions: List of (ground_truth_prefix, predicted_prefix) tuples.
        dataset: PointCloudDataset (for sampling evaluation points).
        tokenizer: PrefixTokenizer.
        n_fit_points: Number of points for constant fitting.

    Returns:
        Dict with 'exact_match', 'token_accuracy', 'algebraic_equiv',
        'mean_r2', 'r2_above_0.9', and per-sample 'details'.
    """
    exact_matches = []
    token_accs = []
    algebraic_matches = []
    r2_scores = []
    details = []

    for i, (gt_str, pred_str) in enumerate(predictions):
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

        # Algebraic equivalence
        algebraic_matches.append(int(equations_equivalent(pred_str, gt_str)))

        # R² via constant fitting
        r2 = None
        if i < len(dataset.samples):
            expr_obj = dataset.samples[i]['expr']
            try:
                with _Timeout(10):
                    pred_expr, constants = prefix_to_sympy(pred_str)
                    cloud = expr_obj.sample(n_fit_points)
                    finite_mask = np.isfinite(cloud).all(axis=1)
                    cloud = cloud[finite_mask]
                    if len(cloud) >= 50:
                        n_vars = len(expr_obj.variables)
                        X = cloud[:, :n_vars]
                        Y = cloud[:, n_vars]
                        var_syms = [sp.Symbol(f'x{j+1}') for j in range(n_vars)]
                        _, _, r2 = fit_constants(pred_expr, constants, X, Y, var_syms)
            except Exception:
                pass

        if r2 is not None:
            r2_scores.append(r2)

        details.append({
            'gt': gt_str, 'pred': pred_str,
            'exact': exact, 'r2': r2,
        })

    n = len(predictions)
    results = {
        'exact_match': np.mean(exact_matches) if exact_matches else 0,
        'token_accuracy': np.mean(token_accs) if token_accs else 0,
        'algebraic_equiv': np.mean(algebraic_matches) if algebraic_matches else 0,
        'mean_r2': np.mean(r2_scores) if r2_scores else float('nan'),
        'median_r2': np.median(r2_scores) if r2_scores else float('nan'),
        'r2_above_0.9': np.mean([r > 0.9 for r in r2_scores]) if r2_scores else 0,
        'n_parseable': len(r2_scores),
        'n_total': n,
        'details': details,
    }
    return results
