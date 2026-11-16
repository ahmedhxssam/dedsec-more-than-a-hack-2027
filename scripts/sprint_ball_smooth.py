"""WPY-position-calibrated smoothing of learned missing-ball residuals.

sigma7 native frames, sqrt endpoint taper and weight1 were selected using
position error only. Never substitutes a measured coordinate or unbounded gap.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scripts.sprint_ball_imputer import features


def predict_smoothed(a,models):
    x,linear,eligible=features(a)
    if not len(x):return linear,eligible
    residual=np.zeros_like(linear)
    residual[eligible]=np.clip(np.stack([m.predict(x) for m in models],axis=1)*20,-25,25)
    smooth=gaussian_filter1d(residual,7,axis=0)
    phase=x[:,14]
    taper=np.sqrt(np.clip(4*phase*(1-phase),0,1))
    linear[eligible]+=smooth[eligible]*taper[:,None]
    return linear,eligible
