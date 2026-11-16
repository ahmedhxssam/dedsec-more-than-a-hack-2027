"""Read supplied Metrica tracking and automatic tags without inventing labels.

All public frame indices are zero based; intervals are half open. Coordinates
remain normalized pitch coordinates and supplied speeds remain unconverted. NaN is
missing evidence, never a dead-ball signal. Team slots are resolved from XML
channel references, not assumed to have the same order between recordings.
The XML declares km/h, but the current export's numeric speeds agree with m/s;
see reports/input_inventory.md. Use derived displacement for physical constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import csv
import warnings
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np


def _parameters(element: ET.Element | None) -> dict[str, str | None]:
    if element is None:
        return {}
    return {p.findtext("Name", ""): p.findtext("Value") for p in element.findall("ProviderParameter")}


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Parse the XML's explicit column order, roles, units, and frame origin.

    These exports describe generic tracks, not authenticated player identities.
    Their provider half boundaries are retained verbatim, not trusted as periods.
    Only the documented x,y,s triplet layout is supported; others fail explicitly.
    """
    root = ET.parse(path).getroot()
    # Permit the same schema with an actual namespace, rather than only xmlns declarations.
    for e in root.iter():
        e.tag = e.tag.rsplit("}", 1)[-1]
    global_config = root.find("Metadata/GlobalConfig")
    session = root.find("Metadata/Sessions/Session")
    fmt = root.find("DataFormatSpecifications/DataFormatSpecification")
    if global_config is None or session is None or fmt is None:
        raise ValueError("Missing Metrica metadata or data format specification")
    if fmt.get("separator") != ":":
        raise ValueError("Only colon-separated Metrica records are supported")
    score = session.find("MatchParameters/Score")
    if score is None:
        raise ValueError("Metadata must explicitly identify local and visiting teams")
    home_id, away_id = score.get("idLocalTeam"), score.get("idVisitingTeam")
    teams = {}
    for t in root.findall("Metadata/Teams/Team"):
        tid = t.attrib["id"]
        teams[tid] = {"id": tid, "name": t.findtext("Name"),
                      "label": "H" if tid == home_id else "A" if tid == away_id else "X",
                      "parameters": _parameters(t.find("ProviderTeamsParameters"))}
    channels = {c.attrib["id"]: dict(c.attrib) for c in root.findall("Metadata/PlayerChannels/PlayerChannel")}
    player_records = {p.attrib["id"]: p for p in root.findall("Metadata/Players/Player")}
    registers = fmt.findall("SplitRegister")
    if len(registers) != 2 or registers[0].get("separator") != ";":
        raise ValueError("Expected player triplets followed by a ball triplet")
    players = []
    for slot, register in enumerate(registers[0].findall("SplitRegister")):
        refs = [channels[r.attrib["playerChannelId"]] for r in register.findall("PlayerChannelRef")]
        if [r["channelId"] for r in refs] != ["x", "y", "s"] or len({r["playerId"] for r in refs}) != 1:
            raise ValueError(f"Unsupported player channel layout at slot {slot}")
        pid = refs[0]["playerId"]
        p = player_records[pid]
        tid = p.attrib["teamId"]
        players.append({"slot": slot, "id": pid, "name": p.findtext("Name"),
                        "shirt_number": p.findtext("ShirtNumber"), "team_id": tid,
                        "team_label": teams[tid]["label"],
                        "parameters": _parameters(p.find("ProviderPlayerParameters"))})
    if [r.get("channelId") for r in registers[1].findall("BallChannelRef")] != ["x", "y", "s"]:
        raise ValueError("Unsupported ball channel layout")
    units = {c.attrib["id"]: c.findtext("Unit") for c in root.findall("Metadata/Devices/Device/Sensors/Sensor/Channels/Channel")}
    if units != {"x": "normalized", "y": "normalized", "s": "kph"}:
        raise ValueError(f"Unsupported units: {units}")
    start = int(fmt.attrib["startFrame"])
    end = int(fmt.attrib["endFrame"])
    return {"source": str(path), "fps": float(global_config.findtext("FrameRate", "nan")),
            "provider": global_config.findtext("ProviderName"), "units": units,
            "source_frame_start": start, "source_frame_end_inclusive": end,
            "n_frames": end - start + 1,
            "pitch_size_m": [float(session.findtext("FieldSize/Width", "nan")), float(session.findtext("FieldSize/Height", "nan"))],
            "teams": teams, "players": players,
            "provider_parameters": _parameters(global_config.find("ProviderGlobalParameters"))}


@dataclass
class TrackingData:
    frame_ids: np.ndarray
    players_xy: np.ndarray
    player_speed: np.ndarray
    ball_xy: np.ndarray
    ball_speed: np.ndarray
    player_team: np.ndarray
    present: np.ndarray
    metadata: dict[str, Any]
    issues: list[dict[str, Any]]

    @property
    def player_speed_kph(self) -> np.ndarray:
        return self.player_speed

    @property
    def ball_speed_kph(self) -> np.ndarray:
        return self.ball_speed


def load_tracking(path: str | Path, metadata: dict[str, Any] | str | Path,
                  n_frames: int | None = None) -> TrackingData:
    """Load all rows, preserving missing rows and measurements as NaN.

    Malformed, duplicate, out-of-order or out-of-range records fail loudly. Gaps
    are recorded and left empty. No interpolation or finite-coordinate clipping
    is performed. The third channel is speed, not detector confidence.
    """
    if not isinstance(metadata, dict):
        metadata = load_metadata(metadata)
    n = metadata["n_frames"] if n_frames is None else int(n_frames)
    if n < 0:
        raise ValueError("n_frames must be nonnegative")
    p = len(metadata["players"])
    values = np.full((n, p + 1, 3), np.nan, dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    previous = -1
    issues: list[dict[str, Any]] = []
    origin = metadata["source_frame_start"]
    with Path(path).open() as f:
        for line_number, line in enumerate(f, 1):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number}: expected frame:players:ball")
            frame = int(parts[0]) - origin
            if frame < 0 or frame >= n or frame <= previous:
                raise ValueError(f"{path}:{line_number}: duplicate/out-of-order/out-of-range frame {frame}")
            if parts[1].count(";") != p - 1:
                raise ValueError(f"{path}:{line_number}: wrong player slot count")
            numeric = parts[1].replace(";", ",") + "," + parts[2]
            if numeric.count(",") != 3 * (p + 1) - 1:
                raise ValueError(f"{path}:{line_number}: wrong channel count")
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                try:
                    row = np.fromstring(numeric, dtype=np.float32, sep=",")
                except (ValueError, DeprecationWarning) as exc:
                    raise ValueError(f"{path}:{line_number}: malformed numeric value") from exc
            if row.size != 3 * (p + 1) or np.isinf(row).any():
                raise ValueError(f"{path}:{line_number}: malformed or infinite measurement")
            if frame > previous + 1:
                issues.append({"kind": "missing_tracking_rows", "start_frame": previous + 1, "end_frame": frame})
            values[frame] = row.reshape(p + 1, 3)
            present[frame] = True
            previous = frame
    if previous < n - 1:
        issues.append({"kind": "missing_tracking_rows", "start_frame": previous + 1, "end_frame": n})
    return TrackingData(np.arange(n, dtype=np.int32), values[:, :p, :2], values[:, :p, 2],
                        values[:, p, :2], values[:, p, 2],
                        np.array([x["team_label"] for x in metadata["players"]]), present, metadata, issues)


@dataclass
class HomographyData:
    matrices: np.ndarray
    valid: np.ndarray
    reprojection_error: np.ndarray
    present: np.ndarray
    issues: list[dict[str, Any]]


def load_homography(path: str | Path, n_frames: int) -> HomographyData:
    """Read zero-based homographies; valid requires flag, finite matrix, nonnegative error.

    These files do not specify the reprojection-error unit. Invalid matrices are
    retained for auditing but must not be used by callers. Missing rows stay NaN.
    """
    h = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
    err = np.full(n_frames, np.nan, dtype=np.float32)
    flag = np.zeros(n_frames, dtype=bool)
    present = np.zeros(n_frames, dtype=bool)
    issues = []
    previous = -1
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        expected = ["frame", *[f"h{i}{j}" for i in range(3) for j in range(3)], "reprojection_error", "valid"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected homography columns: {reader.fieldnames}")
        for row in reader:
            frame = int(row["frame"])
            if frame <= previous or frame < 0 or frame >= n_frames:
                raise ValueError(f"Duplicate/out-of-order/out-of-range homography frame {frame}")
            h[frame] = np.array([float(row[k]) for k in expected[1:10]]).reshape(3, 3)
            err[frame] = float(row["reprojection_error"])
            value = int(row["valid"])
            if value not in (0, 1):
                raise ValueError(f"Invalid valid flag {value} at frame {frame}")
            flag[frame] = value == 1
            present[frame] = True
            previous = frame
    valid = flag & np.isfinite(h).all(axis=(1, 2)) & np.isfinite(err) & (err >= 0)
    if np.any(flag & ~valid):
        issues.append({"kind": "flagged_valid_but_unusable", "count": int(np.sum(flag & ~valid))})
    if not present.all():
        issues.append({"kind": "missing_homography_rows", "count": int(np.sum(~present))})
    return HomographyData(h, valid, err, present, issues)


def project_pitch_to_image(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to normalized pitch xy.

    Direction follows official starter/load_homography.py (the CSV alone omits it).
    Non-finite input or near-zero denominator produces NaN, never a clipped point.
    Supports a single matrix and an arbitrary point array ending in (..., 2).
    """
    xy = np.asarray(xy, dtype=float)
    h = np.asarray(matrix, dtype=float)
    if h.shape != (3, 3) or xy.shape[-1] != 2:
        raise ValueError("Expected one 3x3 matrix and points ending in dimension 2")
    q = np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), axis=-1) @ h.T
    result = np.full_like(xy, np.nan)
    good = np.isfinite(q).all(axis=-1) & (np.abs(q[..., 2]) > 1e-12)
    result[good] = q[good, :2] / q[good, 2:3]
    return result


project_to_pixels = project_pitch_to_image


@dataclass(frozen=True)
class TagInterval:
    id: str
    code: str
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    labels: dict[str, str]
    clipped: bool = False


def load_tags(path: str | Path, fps: float, n_frames: int | None = None) -> list[TagInterval]:
    """Parse automatic event windows, never ground-truth possession intervals.

    Samples at frame/fps in [start,end) are included: ceil(start*fps) through
    ceil(end*fps)-1. Decimal avoids binary rounding off-by-one errors. Windows
    can overlap and can denote defending teams; no possession labels are derived.
    """
    result = []
    rate = Decimal(str(fps))
    if not rate.is_finite() or rate <= 0:
        raise ValueError("fps must be finite and positive")
    for x in ET.parse(path).getroot().findall("ALL_INSTANCES/instance"):
        start = Decimal(x.findtext("start", "NaN"))
        end = Decimal(x.findtext("end", "NaN"))
        if not start.is_finite() or not end.is_finite() or end < start:
            raise ValueError(f"Invalid interval for tag {x.findtext('ID')}")
        s = int((start * rate).to_integral_value(rounding=ROUND_CEILING))
        e = int((end * rate).to_integral_value(rounding=ROUND_CEILING))
        cs, ce = max(0, s), max(0, e)
        if n_frames is not None:
            cs, ce = min(cs, n_frames), min(ce, n_frames)
        labels = {y.findtext("group", ""): y.findtext("text", "") for y in x.findall("label")}
        result.append(TagInterval(x.findtext("ID", ""), x.findtext("code", ""), float(start), float(end), cs, ce, labels, (cs, ce) != (s, e)))
    return sorted(result, key=lambda x: (x.start_frame, x.end_frame, x.id))


def dense_tag_evidence(tags: list[TagInterval], metadata: dict[str, Any], n_frames: int) -> dict[str, np.ndarray]:
    """Convert automatic tactical phases into weak team hints, preserving conflict.

    Columns are H,A. Defending tags identify the defending team and are inverted.
    Complementary attacking/defending windows are correlated, so their maximum
    support is one per team/frame: duplicated tags cannot manufacture confidence.
    Geometric events, goals and set-piece windows do not establish live control.
    """
    attack_codes = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION"}
    defend_codes = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                    "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
    names = {t["name"]: t["label"] for t in metadata["teams"].values()}
    support = np.zeros((n_frames, 2), dtype=np.float32)
    count = np.zeros(n_frames, dtype=np.uint16)
    set_pieces = np.zeros(n_frames, dtype=bool)
    goals = np.zeros(n_frames, dtype=bool)
    for tag in tags:
        s, e = max(0, tag.start_frame), min(n_frames, tag.end_frame)
        if e <= s:
            continue
        count[s:e] += 1
        if tag.code == "SET PIECES":
            set_pieces[s:e] = True
        elif tag.code == "GOALS":
            goals[s:e] = True
        team = names.get(tag.labels.get("Team"))
        if team not in ("H", "A"):
            continue
        idx = 0 if team == "H" else 1
        if tag.code in defend_codes:
            idx = 1 - idx
        elif tag.code not in attack_codes:
            continue
        support[s:e, idx] = 1.0
    return {"team_support": support, "team_conflict": np.all(support > 0, axis=1),
            "tag_count": count, "set_piece_window": set_pieces, "goal_window": goals}
