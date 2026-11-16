"""Scientific source closure shared by complete-baseline inference seals."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scientific_sources():
    names = [
        'second_zero/full_baseline_sources.py', 'second_zero/full_baseline_pipeline.py',
        'scripts/aggressive_unseen_pipeline.py', 'second_zero/common.py', 'second_zero/evaluate.py',
        'second_zero/contextual.py', 'second_zero/temporal_ownership.py', 'second_zero/temporal_live_dead.py',
        'second_zero/sprint_model_owner.py', 'second_zero/sprint_model_interpolation.py',
        'second_zero/overnight_boundaries.py', 'second_zero/joint_controller_ball.py',
        'second_zero/vision_player_temporal.py', 'second_zero/set_ownership.py', 'second_zero/focused_boundaries.py',
        'scripts/sprint_memory.py', 'scripts/sprint_ball_imputer.py', 'scripts/sprint_carrier_specialist.py',
        'scripts/sprint_gap_specialist.py', 'scripts/sprint_status_context.py',
        'scripts/evaluate_temporal_ownership.py', 'scripts/sprint_decoder_segments_geometry_v2.py',
        'scripts/sprint_decoder_lattice.py', 'second_zero/sprint_decoder.py',
        'scripts/train_contextual.py', 'second_zero/global_team_context.py',
        'scripts/global_team_context_probe.py', 'vendor/mtah/evaluation/possession_metrics.py',
    ]
    return [ROOT/name for name in names]
