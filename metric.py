"""Simulator-independent success metrics. Distances and heights are in metres.

These are configurable evaluation criteria, not calibrated paper benchmarks.
Particle correspondence is preserved by Flex, so folding compares particle IDs
against the demonstration goal (no registration that could hide a wrong fold).
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class MetricConfig:
    fold_rmse: float = 0.03
    fold_p95: float = 0.06
    flatten_coverage: float = 0.90
    place_coverage: float = 0.50
    place_inside_fraction: float = 0.99
    hang_min_height: float = 0.10
    hang_support_distance: float = 0.04

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"Invalid metric threshold {name}: {value}")
            if value is not None and ("coverage" in name or "fraction" in name) and value > 1:
                raise ValueError(f"{name} must be in [0, 1]")


def particles(state):
    data = state["particle_pos"] if isinstance(state, dict) else state
    pos = np.asarray(data, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 4)
    if pos.ndim != 2 or pos.shape[1] not in (3, 4) or not len(pos):
        raise ValueError("Expected nonempty N x 3 / N x 4 particle positions")
    if not np.isfinite(pos).all():
        raise ValueError("Particle positions must be finite")
    return pos[:, :3]


def covered_area(state, particle_radius=0.00625):
    """SoftGym's 100 x 100 projected particle coverage, without importing Flex."""
    xy = particles(state)[:, [0, 2]]
    span = np.ptp(xy, axis=0) / 100
    if np.any(span <= 0):
        return 0.0
    offset = xy - xy.min(axis=0)
    low = np.maximum(np.round((offset - particle_radius) / span).astype(int), 0)
    high = np.minimum(np.round((offset + particle_radius) / span).astype(int), 100)
    # Match vectorized_range/vectorized_meshgrid in flex_utils exactly, since
    # the cache's max_area was computed using that discretization.
    grid = np.zeros(10000, dtype=bool)
    ranges = []
    for axis in (0, 1):
        n = int(np.max(high[:, axis] - low[:, axis])) + 1
        ranges.append(np.floor(np.arange(n) * (high[:, axis] - low[:, axis])[:, None] / n
                               + low[:, axis, None]).astype(int))
    grid[np.clip((ranges[0][:, :, None] * 100 + ranges[1][:, None, :]).ravel(), 0, 9999)] = True
    return float(grid.sum() * np.prod(span))


def rigid_frame(rigid_states):
    """Return center and horizontal local axes from the generator's five shapes."""
    shapes = np.asarray(rigid_states, dtype=float).reshape(-1, 14)
    if len(shapes) != 5 or not np.isfinite(shapes).all():
        raise ValueError("Expected five finite box/hanger shape states")
    axis = (shapes[2, :3] - shapes[1, :3])[[0, 2]]
    width = np.linalg.norm(axis)
    if width <= 0:
        raise ValueError("Degenerate rigid object")
    axis /= width
    normal = np.array([-axis[1], axis[0]])
    return shapes, axis, normal, width


def evaluate(task, current_state, goal_state=None, cloth_type="Towel",
             rigid_states=None, config=None, released=True, particle_radius=0.00625,
             single_step=None):
    """Return a JSON-serializable success decision and its measured quantities.

    Placement requires particle containment and coverage; hanging requires all
    particles off the table, particles near the bar, and cloth on both sides.
    Every task requires released grippers. Call after allowing the simulation
    to relax; convergence of the settling diagnostic is not a success criterion.
    """
    cfg = config or MetricConfig()
    pos = particles(current_state)
    kind = next((k for k in ("fold", "flatten", "place", "hang") if task.lower().endswith(k)), None)
    result = {"task": task, "released": bool(released), "thresholds": asdict(cfg)}
    if kind == "fold":
        if not isinstance(single_step, bool):
            raise ValueError("Folding evaluation requires boolean single_step metadata")
        if goal_state is None:
            raise ValueError("Folding requires a demonstration goal state")
        goal = particles(goal_state)
        if pos.shape != goal.shape:
            raise ValueError("Current and goal particle correspondence differs")
        errors = np.linalg.norm(pos - goal, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        p95 = float(np.percentile(errors, 95))
        # Match the task's execution mode, not the number of actions predicted
        # by the planner (which may be incomplete or include retries).
        multiplier = 1 if single_step else 2
        rmse_threshold, p95_threshold = cfg.fold_rmse * multiplier, cfg.fold_p95 * multiplier
        result["thresholds"].update(fold_rmse=rmse_threshold, fold_p95=p95_threshold)
        result.update(single_step=single_step, particle_rmse=rmse, particle_p95=p95,
                      fold_threshold_multiplier=multiplier)
        success = rmse <= rmse_threshold and p95 <= p95_threshold
    elif kind in ("flatten", "place"):
        max_area = float(current_state["max_area"])
        if not np.isfinite(max_area) or max_area <= 0:
            raise ValueError("max_area must be finite and positive")
        coverage = covered_area(pos, particle_radius) / max_area
        threshold = cfg.place_coverage if kind == "place" else cfg.flatten_coverage
        result.update(coverage=float(coverage), coverage_threshold=threshold)
        success = coverage >= threshold
        if kind == "place":
            shapes, axis, normal, width = rigid_frame(rigid_states)
            local = pos[:, [0, 2]] - shapes[0, [0, 2]]
            depth = abs(np.dot(shapes[4, [0, 2]] - shapes[3, [0, 2]], normal))
            # Walls in the existing generators have 2 cm thickness.
            inside = ((abs(local @ axis) <= width / 2 - 0.01) &
                      (abs(local @ normal) <= depth / 2 - 0.01) &
                      (pos[:, 1] >= shapes[0, 1]) &
                      (pos[:, 1] <= 2 * shapes[1, 1] + 0.02))
            fraction = float(inside.mean())
            result["inside_fraction"] = fraction
            success = success and fraction >= cfg.place_inside_fraction
    elif kind == "hang":
        shapes, axis, normal, width = rigid_frame(rigid_states)
        bar = shapes[0, :3]
        local = pos[:, [0, 2]] - bar[[0, 2]]
        along, across = local @ axis, local @ normal
        distance = np.sqrt(np.maximum(abs(along) - width / 2, 0) ** 2 +
                           across ** 2 + (pos[:, 1] - bar[1]) ** 2)
        support_distance = float(distance.min())
        draped = bool(np.any(across < -particle_radius) and np.any(across > particle_radius))
        result.update(min_height=float(pos[:, 1].min()), support_distance=support_distance, draped=draped)
        success = (pos[:, 1].min() > cfg.hang_min_height and
                   support_distance <= cfg.hang_support_distance and draped)
    else:
        raise ValueError(f"Unsupported task: {task}")
    result["success"] = bool(success and released)
    return result


def is_success(*args, **kwargs):
    return evaluate(*args, **kwargs)["success"]
