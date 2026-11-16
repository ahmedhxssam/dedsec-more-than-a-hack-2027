"""Fixed parent/native4 alpha .5, q .98, typed scale1; no source-NN branch."""
import numpy as np
from scripts import wqq_campaign_status_mass_v1 as status
from scripts import wqq_campaign_transition_scale_v1 as transition

POLICY = dict(id="strong_scale1", alpha=.5, parent_confidence=.98,
              status_mode="joint", transition_scale=1., apply_source=False)


def compose_logits(parent, native):
    if status.Q != POLICY["parent_confidence"] or transition.Q != POLICY["parent_confidence"]:
        raise ValueError("The original parent confidence changed")
    nodes = status.mixed_nodes(parent, native, POLICY["alpha"])[0]
    return transition.decode(nodes, np.asarray(native, np.float32)[:, 2:], 1.)


def predict_raw(raw, bundle):
    """Only raw ball/players/team/pitch and the packaged original bundle enter."""
    from scripts.wqq_campaign_raw_replay_v2 import raw_identity
    before = raw_identity(raw)
    parent, intermediates, trace = bundle.parent_from_raw(raw)
    native = bundle.native_from_raw(raw)
    predicted = compose_logits(parent, native)
    if raw_identity(raw) != before:
        raise ValueError("Strong-scale1 inference mutated the observations")
    return dict(predicted=predicted, parent=parent, logits=native), trace
