"""Anonymous, age-aware spatial game memory from observations and predictions.

No possession references, stable identities, true player counts, or hidden ball
positions are inputs. Bilinear maximum occupancy is duplicate-insensitive; a
cell is a spatial observation bank, never a player identity. Aged evidence keeps
its uncertainty and can be contradicted by current observations or predictions.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit

VERSION = 'anonymous-context-memory-v1'
GRID = (8, 5)
HORIZON = 250
FPS = 25.
CHUNK = 1024
TEAM_NAMES = [
    'occupancy_mass', 'mean_age', 'maximum_age', 'mean_uncertainty',
    'velocity_support', 'moving_support', 'mean_speed', 'flow_coherence',
    'absolute_center_x', 'absolute_center_y', 'spread_x', 'spread_y',
    'endline_band_mass', 'touchline_band_mass',
    'stationary_ball_overlap3', 'stationary_ball_overlap9',
    'advected_ball_overlap3', 'advected_ball_overlap9',
    'mean_ball_distance', 'mean_approach_speed', 'ball_velocity_alignment']
STATIC_GLOBAL_NAMES = ['ball_observed', 'ball_usable', 'ball_velocity_support',
    'ball_speed', 'absolute_ball_x', 'absolute_ball_y', 'signed_owner_logit',
    'owner_probability', 'owner_confidence', 'predicted_live']
MEMORY_GLOBAL_NAMES = ['ball_anchor_valid', 'ball_anchor_age', 'ball_uncertainty',
    'ball_anchor_speed', 'absolute_ball_anchor_x', 'absolute_ball_anchor_y',
    'absolute_ball_forecast_x', 'absolute_ball_forecast_y', 'owner_anchor_valid',
    'owner_anchor_age', 'signed_owner_anchor_logit', 'owner_anchor_probability',
    'owner_anchor_confidence', 'owner_anchor_strength', 'owner_current_agreement',
    'ball_current_observed']


def _motion(positions, speed_limit):
    """Local adjacent observed displacements only; no gap-spanning association."""
    p = np.asarray(positions, np.float32)
    seen = np.isfinite(p).all(-1)
    delta = np.diff(p, axis=0) * FPS
    valid = seen[1:] & seen[:-1] & (np.linalg.norm(delta, axis=-1) <= speed_limit)
    delta = np.where(valid[..., None], delta, 0.)
    value = np.zeros_like(p); count = np.zeros(p.shape[:-1], np.float32)
    value[1:] += delta; value[:-1] += delta
    count[1:] += valid; count[:-1] += valid
    value /= np.maximum(count[..., None], 1.)
    return value, count > 0


def _deposits(players, team, velocity, velocity_valid, pitch):
    """Chunk-local deduplicated maximum splats, at most O(chunk*players)."""
    n, p = players.shape[:2]; cells = GRID[0] * GRID[1]; total = n * 2 * cells
    valid = np.isfinite(players).all(-1)
    valid &= (players[..., 0] >= 0) & (players[..., 0] <= pitch[0])
    valid &= (players[..., 1] >= 0) & (players[..., 1] <= pitch[1])
    frame, slot = np.nonzero(valid)
    out = np.zeros((n, 2 * cells, 7), np.float32)
    if not len(frame):
        return out
    xy = players[frame, slot]; vv = velocity[frame, slot]; ok = velocity_valid[frame, slot]
    # Exact repeated observations carry no extra mass or centroid vote. Distinct
    # coincident motions remain ambiguous evidence, not an inferred identity.
    records = np.column_stack([frame, team[slot], xy, vv, ok])
    _, unique = np.unique(records, axis=0, return_index=True)
    frame, slot, xy, vv, ok = frame[unique], slot[unique], xy[unique], vv[unique], ok[unique]
    coordinate = xy / pitch * np.array([GRID[0] - 1, GRID[1] - 1])
    low = np.floor(coordinate).astype(int); fraction = coordinate - low
    high = np.minimum(low + 1, np.array(GRID) - 1)
    indices = []; supports = []; origins = []
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        gx = (low if a == 0 else high)[:, 0]; gy = (low if b == 0 else high)[:, 1]
        w = (1 - fraction[:, 0] if a == 0 else fraction[:, 0]) * (1 - fraction[:, 1] if b == 0 else fraction[:, 1])
        keep = w > 1e-7
        indices.append(((frame * 2 + team[slot]) * cells + gx * GRID[1] + gy)[keep])
        supports.append(w[keep]); origins.append(np.flatnonzero(keep))
    index = np.concatenate(indices); weight = np.concatenate(supports); origin = np.concatenate(origins)
    maximum = np.zeros(total, np.float64); np.maximum.at(maximum, index, weight)
    selected = weight >= maximum[index] - 1e-9
    index, origin = index[selected], origin[selected]
    count = np.bincount(index, minlength=total); denominator = np.maximum(count, 1.)
    values = np.c_[xy, vv, ok, np.linalg.norm(vv, axis=1) > .1]
    out[..., 0] = maximum.reshape(n, 2 * cells)
    for j in range(6):
        out[..., j + 1] = (np.bincount(index, weights=values[origin, j], minlength=total) / denominator).reshape(n, 2 * cells)
    return out


def _summaries(support, position, velocity, velocity_valid, moving, age, sigma,
               ball_anchor, ball_forecast, ball_velocity, ball_sigma, ball_valid, pitch):
    n = len(support); cells = GRID[0] * GRID[1]
    w = support.reshape(n, 2, cells); den = np.maximum(w.sum(2), 1e-8)
    pos = position.reshape(n, 2, cells, 2); vel = velocity.reshape(n, 2, cells, 2)
    mean = lambda a: (w * a.reshape(n, 2, cells)).sum(2) / den
    center = (w[..., None] * pos).sum(2) / den[..., None]
    centered = (pos - center[:, :, None]) / (pitch / 2)
    spread = np.sqrt(np.maximum((w[..., None] * centered ** 2).sum(2) / den[..., None], 0))
    mean_v = (w[..., None] * vel).sum(2) / den[..., None]
    speeds = np.linalg.norm(vel, axis=-1); speed = (w * speeds).sum(2) / den
    fields = [w.sum(2) / cells, mean(age) / HORIZON,
        np.where(w > 0, age.reshape(n, 2, cells), 0).max(2) / HORIZON,
        mean(sigma) / 30, mean(velocity_valid), mean(moving), speed / 12,
        np.linalg.norm(mean_v, axis=-1) / np.maximum(speed, .1),
        abs(center[..., 0] - pitch[0] / 2) / (pitch[0] / 2),
        abs(center[..., 1] - pitch[1] / 2) / (pitch[1] / 2), spread[..., 0], spread[..., 1],
        (w * (np.minimum(pos[..., 0], pitch[0] - pos[..., 0]) < pitch[0] * .2)).sum(2) / den,
        (w * (np.minimum(pos[..., 1], pitch[1] - pos[..., 1]) < pitch[1] * .2)).sum(2) / den]
    p_sigma = sigma.reshape(n, 2, cells)
    for query in (ball_anchor, ball_forecast):
        distance2 = ((pos - query[:, None, None]) ** 2).sum(-1)
        for radius in (3., 9.):
            variance = radius ** 2 + p_sigma ** 2 + ball_sigma[:, None, None] ** 2
            fields.append((w * np.exp(-distance2 / (2 * variance))).sum(2) / den * ball_valid[:, None])
    delta = ball_forecast[:, None, None] - pos
    distance = np.linalg.norm(delta, axis=-1)
    fields.extend([(w * distance).sum(2) / den / 70 * ball_valid[:, None],
        (w * (delta * vel).sum(-1) / np.maximum(distance, 1.)).sum(2) / den / 12 * ball_valid[:, None],
        (w * (vel * ball_velocity[:, None, None]).sum(-1)).sum(2) / den / (12 * 60) * ball_valid[:, None]])
    result = np.stack(fields, axis=-1)
    result[w.sum(2) == 0] = 0
    return result.astype(np.float32)


def _last_anchor(observed, live):
    ix = np.arange(len(live)); previous = np.maximum.accumulate(np.where(observed & live, ix, -HORIZON - 1))
    reset = np.maximum.accumulate(np.where(~live, ix, -HORIZON - 2))
    age = ix - previous
    valid = (previous > reset) & (age <= HORIZON) & (previous >= 0)
    return np.clip(previous, 0, max(0, len(ix) - 1)), age, valid


def _pass(raw, owner, status, reverse, make_static):
    pitch = np.asarray(raw.get('pitch', (105.3, 68)), np.float32)
    players = np.asarray(raw['players'], np.float32); ball = np.asarray(raw['ball'], np.float32)
    team = np.asarray(raw['team'], np.int64); n = len(ball)
    pv, pvalid = _motion(players, 12.); bv, bvalid = _motion(ball, 60.)
    if reverse:
        players, ball, owner, status = players[::-1], ball[::-1], owner[::-1], status[::-1]
        pv, pvalid, bv, bvalid = -pv[::-1], pvalid[::-1], -bv[::-1], bvalid[::-1]
    live = status < 2; seen = np.isfinite(ball).all(1)
    usable = seen & (ball[:, 0] >= -10) & (ball[:, 0] <= pitch[0] + 10) & (ball[:, 1] >= -10) & (ball[:, 1] <= pitch[1] + 10)
    at, ball_age, ball_ok = _last_anchor(usable, live)
    safe_ball = np.nan_to_num(ball, nan=0, posinf=0, neginf=0)
    anchor = safe_ball[at]; bvelocity = bv[at] * bvalid[at, None]
    seconds = np.minimum(ball_age, HORIZON) / FPS
    forecast = anchor + bvelocity * (1 - np.exp(-seconds))[:, None]
    b_sigma = 1.5 + 2 * seconds + .15 * np.linalg.norm(bvelocity, axis=1) * seconds
    probability = expit(owner); confidence = 2 * abs(probability - .5)
    strong = (abs(owner) >= np.log(4.)) & live
    count = np.r_[0, np.cumsum((strong & (owner >= 0)).astype(int))]
    pos5 = count[1:] - count[np.maximum(0, np.arange(n) - 4)]
    count = np.r_[0, np.cumsum((strong & (owner < 0)).astype(int))]
    neg5 = count[1:] - count[np.maximum(0, np.arange(n) - 4)]
    owner_at, owner_age, owner_ok = _last_anchor(((pos5 == 5) | (neg5 == 5)) & usable, live)
    owner_anchor = owner[owner_at]; owner_confidence = confidence[owner_at]
    memory = np.zeros((n, 2, 2 * len(TEAM_NAMES) + len(MEMORY_GLOBAL_NAMES)), np.float32)
    static = np.zeros((n, 2, 2 * len(TEAM_NAMES) + len(STATIC_GLOBAL_NAMES)), np.float32) if make_static else None
    cells = 2 * GRID[0] * GRID[1]
    bank = np.zeros((cells, 7), np.float32); stamp = np.full(cells, -HORIZON - 1, np.int64); last_reset = -HORIZON - 2
    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK); length = stop - start; sl = slice(start, stop)
        observed = _deposits(players[sl], team, pv[sl], pvalid[sl], pitch)
        if make_static:
            now = _summaries(observed[..., 0], observed[..., 1:3], observed[..., 3:5], observed[..., 5], observed[..., 6], np.zeros((length, cells)), np.ones((length, cells)), safe_ball[sl], safe_ball[sl], bv[sl], np.full(length, 1.5), usable[sl], pitch)
            for perspective in (0, 1):
                sign = 2 * perspective - 1
                global_now = np.c_[seen[sl], usable[sl], bvalid[sl], np.linalg.norm(bv[sl], axis=1) / 60,
                    abs(safe_ball[sl, 0] - pitch[0] / 2) / (pitch[0] / 2) * usable[sl],
                    abs(safe_ball[sl, 1] - pitch[1] / 2) / (pitch[1] / 2) * usable[sl],
                    sign * owner[sl] / 8, expit(sign * owner[sl]), confidence[sl], live[sl]]
                static[sl, perspective] = np.c_[now[:, perspective], now[:, 1 - perspective], global_now]
        local = np.arange(1, length + 1)[:, None]
        updated = np.maximum.accumulate(np.where((observed[..., 0] > 0) & live[sl, None], local, 0), axis=0)
        values = np.concatenate([bank[None], observed], axis=0)
        remembered = values[updated, np.arange(cells)[None]]
        time = np.where(updated > 0, start + updated - 1, stamp[None])
        age = np.arange(start, stop)[:, None] - time
        resets = np.maximum.accumulate(np.r_[last_reset, np.where(~live[sl], np.arange(start, stop), last_reset)])[1:]
        valid = (time > resets[:, None]) & (age <= HORIZON) & (time >= 0)
        elapsed = np.minimum(age, HORIZON) / FPS
        support = remembered[..., 0] * np.exp(-elapsed / 3.) * valid
        position = remembered[..., 1:3] + remembered[..., 3:5] * (2 * (1 - np.exp(-elapsed / 2.)))[..., None]
        sigma = 1 + 1.5 * elapsed
        summary = _summaries(support, position, remembered[..., 3:5], remembered[..., 5], remembered[..., 6], age, sigma,
            anchor[sl], forecast[sl], bvelocity[sl], b_sigma[sl], ball_ok[sl], pitch)
        for perspective in (0, 1):
            sign = 2 * perspective - 1; oa = owner_anchor[sl]; ok = owner_ok[sl]
            global_memory = np.c_[ball_ok[sl], np.minimum(ball_age[sl], HORIZON) / HORIZON * ball_ok[sl],
                b_sigma[sl] / 30 * ball_ok[sl], np.linalg.norm(bvelocity[sl], axis=1) / 60 * ball_ok[sl],
                abs(anchor[sl, 0] - pitch[0] / 2) / (pitch[0] / 2) * ball_ok[sl],
                abs(anchor[sl, 1] - pitch[1] / 2) / (pitch[1] / 2) * ball_ok[sl],
                abs(forecast[sl, 0] - pitch[0] / 2) / (pitch[0] / 2) * ball_ok[sl],
                abs(forecast[sl, 1] - pitch[1] / 2) / (pitch[1] / 2) * ball_ok[sl], ok,
                np.minimum(owner_age[sl], HORIZON) / HORIZON * ok, sign * oa / 8 * ok,
                expit(sign * oa) * ok, owner_confidence[sl] * ok,
                owner_confidence[sl] * np.exp(-np.minimum(owner_age[sl], HORIZON) / (2 * FPS)) * ok,
                (np.sign(oa) == np.sign(owner[sl])) * ok, seen[sl]]
            memory[sl, perspective] = np.c_[summary[:, perspective], summary[:, 1 - perspective], global_memory]
        memory[sl][~live[sl]] = 0
        bank = remembered[-1].copy(); stamp = time[-1].copy(); last_reset = int(resets[-1])
    if reverse:
        memory = memory[::-1].copy()
    return static, memory


def features(raw, owner_logit, status):
    """Return static[N,2,52], memory[N,2,116], metadata; labels are not accepted."""
    owner = np.asarray(owner_logit, np.float32).reshape(-1); state = np.asarray(status).reshape(-1)
    n = len(raw['ball'])
    if not n or len(owner) != n or len(state) != n or len(raw['players']) != n:
        raise ValueError('Expected nonempty aligned native-frame observations and predicted state')
    if set(np.unique(raw['team'])) != {0, 1} or not np.isfinite(owner).all():
        raise ValueError('Expected two mapped teams and finite owner logits')
    static, past = _pass(raw, owner, state, False, True)
    _, future = _pass(raw, owner, state, True, False)
    memory = np.concatenate([past, future], axis=-1)
    static = np.clip(np.nan_to_num(static), -5, 5).astype(np.float32)
    memory = np.clip(np.nan_to_num(memory), -5, 5).astype(np.float32)
    relative_names = ['own_' + s for s in TEAM_NAMES] + ['opponent_' + s for s in TEAM_NAMES]
    metadata = {'version': VERSION, 'static_feature_names': relative_names + STATIC_GLOBAL_NAMES,
        'memory_feature_names': [direction + '_' + s for direction in ('past', 'future') for s in relative_names + MEMORY_GLOBAL_NAMES],
        'static_channels': static.shape[-1], 'memory_channels': memory.shape[-1], 'grid_nodes': list(GRID),
        'native_fps': FPS, 'age_cap_frames': HORIZON, 'chunk_frames': CHUNK,
        'player_memory': 'Anonymous maximum bilinear occupancy, exact duplicate observation tuples removed; latest node centroid/local motion persists with age, damping and increasing uncertainty. Occupancy is not a player count.',
        'velocity': 'Adjacent measured displacements only, up to12m/s player or60m/s ball; no velocity across a missing endpoint. Stationary positions remain valid.',
        'owner_anchor': 'Five consecutive same-sign logits with probability>=.8 plus measured ball; input predicted ownership only; age decay, no forced label.',
        'resets': 'Every predicted D/X frame clears support; forward/backward banks separate; expiry250frames.',
        'symmetry': 'Shared perspective channels; geometry uses reflection-invariant distances, moments, bands and dot products. No absolute team/side or slot identifier.',
        'labels_or_hidden_positions_accepted': False}
    assert len(metadata['static_feature_names']) == static.shape[-1]
    assert len(metadata['memory_feature_names']) == memory.shape[-1]
    return static, memory, metadata
