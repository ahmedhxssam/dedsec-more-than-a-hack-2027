"""Team-symmetric change evidence, applied only at editable gap boundaries."""
import numpy as np
import torch


@torch.no_grad()
def boundary_logits(model, x, prior, n):
    model.eval()
    _, b, _ = model(torch.from_numpy(x), torch.from_numpy(np.stack([prior, -prior])))
    # A change is invariant to exchanging the two ownership perspectives.
    b = (.5 * (b[0] + b[1])).cpu().numpy()
    return np.interp(np.arange(n), np.arange(len(b))*5, b).astype(np.float32)


def endpoint_credit(arcs, parent, support, boundary, beta, offset):
    a = np.asarray(arcs, np.int64)
    s = a[:, 0]
    internal = (s > 0) & (np.asarray(parent)[np.maximum(s-1, 0)] < 2)
    return beta * np.asarray(support)[s] * internal * (np.asarray(boundary)[s]-offset)
