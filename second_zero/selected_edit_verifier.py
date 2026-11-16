"""A veto-only verifier of already selected, count-preserving boundaries."""
import numpy as np
from second_zero import full_pipeline_selector as selector

ALPHA = 20.
FEATURE_NAMES = [
    'proposal_score', 'log_window', 'log_shift', 'changed_fraction',
    'missing_fraction', 'owner_disagreement', 'baseline_quality_delta',
    'rich_quality_delta', 'unary_delta', 'baseline_mean_delta',
    'baseline_min_delta', 'baseline_log_delta', 'rich_mean_delta',
    'rich_min_delta', 'rich_log_delta', 'shift_over_left', 'shift_over_right',
]


def is_boundary(edit):
    """Exactly one H/A boundary moves; both outside endpoints stay fixed."""
    a, b = edit.old, edit.new
    return (edit.kind == 2 and a.shape == b.shape == (2, 3)
            and np.array_equal(a[:, 2], b[:, 2]) and set(a[:, 2]) == {0, 1}
            and a[0, 0] == b[0, 0] == edit.start
            and a[1, 1] == b[1, 1] == edit.stop
            and a[0, 1] == a[1, 0] and b[0, 1] == b[1, 0]
            and edit.start < b[0, 1] < edit.stop
            and a[0, 1] != b[0, 1])


def select_boundaries(edits, proxy):
    valid = np.array([is_boundary(e) for e in edits], bool)
    utility = np.where(valid & np.isfinite(proxy), proxy, -np.inf)
    return np.asarray(selector.choose(edits, utility), np.int64)


def features(full_features, proxy, edits, indices):
    indices = np.asarray(indices, np.int64)
    if not len(indices):
        return np.empty((0, len(FEATURE_NAMES)), np.float64)
    z = full_features[indices, 700:726].astype(np.float64)
    shifts = np.array([abs(edits[i].new[0, 1] - edits[i].old[0, 1]) for i in indices])
    left = np.array([edits[i].old[0, 1] - edits[i].start for i in indices])
    right = np.array([edits[i].stop - edits[i].old[0, 1] for i in indices])
    x = np.c_[proxy[indices], z[:, [2, 3, 7, 8, 9, 11, 12, 13]],
              z[:, 17:20] - z[:, 14:17], z[:, 23:26] - z[:, 20:23],
              shifts / left, shifts / right]
    if x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid selected-boundary verifier features')
    return x


class GainVerifier:
    """Small regularized gain regressor; never fits unselected proposals."""
    def fit(self, x, y, weights):
        x, y, w = map(lambda a: np.asarray(a, np.float64), (x, y, weights))
        if not len(x) or not all(np.isfinite(a).all() for a in [x, y, w]) or np.any(w <= 0):
            raise ValueError('Invalid verifier fitting rows')
        self.mean = np.sum(x*w[:,None],axis=0)/w.sum()
        self.scale = np.sqrt(np.sum((x-self.mean)**2*w[:,None],axis=0)/w.sum())
        self.scale[self.scale < 1e-10] = 1.
        z = (x-self.mean)/self.scale
        self.intercept = float(np.sum(w*y)/w.sum())
        # Small explicit float64 reductions avoid the local runtime's unstable
        # BLAS matmul path while implementing the weighted ridge equations.
        gram = np.einsum('ni,nj,n->ij',z,z,w,optimize=False)
        rhs = np.sum(z*(w*(y-self.intercept))[:,None],axis=0)
        self.coef = np.linalg.solve(gram+ALPHA*np.eye(x.shape[1]),rhs)
        if not np.isfinite(self.coef).all():raise ValueError('Nonfinite verifier weights')
        return self

    def predict(self, x):
        return np.sum(((x-self.mean)/self.scale)*self.coef,axis=1)+self.intercept if len(x) else np.empty(0)


def calibrate(predicted, targets, weights):
    """Analytic affine calibration on selected WPY actions; no F1 search."""
    u, y, w = map(lambda x: np.asarray(x, np.float64), (predicted, targets, weights))
    valid = np.isfinite(u) & np.isfinite(y) & (w > 0)
    u, y, w = u[valid], y[valid], w[valid]
    if not len(y):
        return {'slope': 0., 'intercept': 0., 'rows': 0}
    w /= w.sum()
    mu, my = np.sum(w * u), np.sum(w * y)
    slope = float(np.clip(np.sum(w * (u-mu) * (y-my)) / max(np.sum(w * (u-mu)**2), 1e-12), 0, 5))
    return {'slope': slope, 'intercept': float(my-slope*mu), 'rows': len(y)}


def accepted(predicted, calibration):
    return np.flatnonzero(calibration['slope'] * np.asarray(predicted) + calibration['intercept'] > 0)


def apply(parent, edits, selected):
    if not all(is_boundary(edits[int(i)]) for i in selected):
        raise ValueError('Verifier may only apply count-preserving boundary shifts')
    out = selector.apply(parent, edits, selected)
    a, b = selector.gs.spells(parent), selector.gs.spells(out)
    if len(a) != len(b) or not np.array_equal(a[:, 2], b[:, 2]):
        raise ValueError('Boundary correction changed spell count or team order')
    return out
