"""Shared testcase IO and Flex scene lifecycle for generation and inference."""
import copy
from pathlib import Path
import pickle
import numpy as np

from metric import rigid_frame


ROOT = Path(__file__).resolve().parent
RIGID_DIMENSIONS = {"place": (0.5, 0.5, 0.05, 0.02), "hang": (0.5, 0.38, 0.30, 0.02)}


def execution_mode(case):
    """Read the number-of-manipulations mode from testcase metadata."""
    value = case.get("single_step")
    if not isinstance(value, bool):
        raise ValueError("Invalid single_step metadata; expected a boolean")
    return "single_step" if value else "multi_step"


def initial_state_type(case):
    """Read the initial-state family from testcase metadata."""
    value = case.get("flat")
    if not isinstance(value, bool):
        raise ValueError("Invalid flat metadata; expected a boolean")
    return "flat" if value else "crumpled"


def task_kind(task):
    for kind in ("fold", "flatten", "hang", "place"):
        if task.lower().endswith(kind):
            return kind
    raise ValueError(f"Unsupported task: {task}")


def cloth_type_for_task(task):
    for cloth in ("Towel", "Tshirt", "Trousers", "Skirt"):
        if task.startswith(cloth):
            return cloth
    raise ValueError(f"Cannot infer cloth type from {task}")


def motion_options(task, cloth_type):
    """Return the execution settings used by the retained evaluation cases."""
    expected_cloth = cloth_type_for_task(task)
    if cloth_type != expected_cloth:
        raise ValueError(f"{task} requires {expected_cloth}, got {cloth_type}")
    kind = task_kind(task)
    speed = 0.006 if task == "TowelFold" else 0.005
    if kind in ("place", "hang") or (cloth_type == "Towel" and kind == "flatten"):
        lift_height = 0.1
    elif cloth_type in ("Trousers", "Skirt"):
        lift_height = 0.15
    else:
        lift_height = 0.125
    return {"pick_speed": speed, "move_speed": speed,
            "place_speed": 0.005, "lift_height": lift_height}


def snapshot(env):
    from softgym.envs.flex_utils import get_state
    state = get_state(env.camera_params)
    state["max_area"] = env.max_area
    return copy.deepcopy(state)


def settle(env, max_steps=300, velocity_threshold=0.2, consecutive=10):
    import pyflex
    stable = 0
    for _ in range(max_steps):
        env.step_simulation()
        velocity = np.asarray(pyflex.get_velocities())
        if not np.isfinite(velocity).all():
            raise RuntimeError("Simulation produced non-finite velocities")
        stable = stable + 1 if np.all(abs(velocity) < velocity_threshold) else 0
        if stable >= consecutive:
            return True
    return False


def restore_scene(env, config, state, task, rigid_states=None):
    """Allocate collision shapes before restoring their saved poses.

    Cached states have zero or two picker shapes; place/hang testcases have
    seven shapes. Passing all seven to a freshly created cloth scene is invalid.
    """
    import pyflex
    from softgym.envs.flex_utils import set_state
    initial = copy.deepcopy(state)
    initial["shape_pos"] = np.empty(0)
    env.reset(copy.deepcopy(config), initial)
    env.set_grasp(False)
    env.movep(env.reset_pos, speed=0.5, min_steps=1)
    if rigid_states is not None:
        kind = task_kind(task)
        if kind not in RIGID_DIMENSIONS:
            raise ValueError("Rigid object supplied for a non-rigid task")
        rigid_frame(rigid_states)
        create = env.create_glass if kind == "place" else env.create_hanger
        create(*RIGID_DIMENSIONS[kind])
    restored = copy.deepcopy(state)
    shapes = np.asarray(state["shape_pos"]).reshape(-1, 14)
    live_shapes = np.asarray(pyflex.get_shape_states()).reshape(-1, 14)
    if rigid_states is not None:
        live_shapes[2:] = np.asarray(rigid_states).reshape(5, 14)
    if len(shapes) == len(live_shapes):
        live_shapes[:2] = shapes[:2]
    restored["shape_pos"] = live_shapes
    set_state(restored)
    env.action_tool.particle_inv_mass = np.asarray(state["particle_pos"]).reshape(-1, 4)[:, 3].copy()
    env.action_tool.picked_particles = [None, None]
    return env


def load_cases(path):
    """Load <task>/<index>_info.pkl and legacy <task>/<index>/info.pkl files.

    Each file is one case: use its first initial state and last goal state.
    Legacy intermediate entries do not become additional evaluation cases.
    """
    path = Path(path)
    def location(source):
        if source.name.endswith("_info.pkl") and source.name[:-9].isdigit():
            return source.parent, source.name[:-9]
        return source.parent.parent, source.parent.name

    paths = [path] if path.is_file() else [
        source for source in path.rglob("*.pkl")
        if source.name == "info.pkl" or
        (source.name.endswith("_info.pkl") and source.name[:-9].isdigit())]
    paths.sort(key=lambda source: (str(location(source)[0]),
                                   int(location(source)[1]) if location(source)[1].isdigit() else -1,
                                   str(source)))
    if not paths:
        raise ValueError(f"No *_info.pkl or legacy info.pkl testcases found in {path}")
    for source in paths:
        with source.open("rb") as stream:
            data = pickle.load(stream)
        task_directory, episode_index = location(source)
        task = data.get("task", task_directory.name)
        cloth = data.get("cloth_type", cloth_type_for_task(task))
        for key in ("single_step", "flat"):
            if not isinstance(data.get(key), bool):
                raise ValueError(f"Missing or invalid boolean {key} metadata: {source}")
        fields = ("instruction", "current_states", "current_configs", "goal_states", "goal_configs", "cloth_3d_list")
        n = len(data["instruction"])
        if not n or any(len(data[key]) != n for key in fields):
            raise ValueError(f"Misaligned testcase lists: {source}")
        for key in ("box_states", "keypoint_indices", "initial_state_types"):
            if key in data and len(data[key]) != n:
                raise ValueError(f"Misaligned {key}: {source}")
        rigid = data.get("box_states", [None])[0]
        if task_kind(task) in ("place", "hang") and rigid is None:
            raise ValueError(f"Missing rigid object states: {source}")
        legacy_start = data.get("initial_state_types", [None])[0]
        if legacy_start is not None and (legacy_start == "flat") != data["flat"]:
            raise ValueError(f"Conflicting flat and initial_state_types metadata: {source}")
        yield {"id": f"{task}/{episode_index}/0", "source": str(source),
               "task": task, "cloth_type": cloth, "instruction": data["instruction"][0],
               "state": data["current_states"][0], "config": data["current_configs"][0],
               "goal_state": data["goal_states"][-1], "cloth3d": data["cloth_3d_list"][0],
               "rigid_states": rigid, "single_step": data["single_step"], "flat": data["flat"],
               "initial_state_type": initial_state_type(data)}


def pixel_to_world(pixel, depth, camera_params, mask=None, max_snap_pixels=20):
    """Backproject (x, y) using depth[y, x]; optionally snap to a nearby mask."""
    from utils.visual import get_world_coord_from_pixel
    x, y = np.asarray(pixel, dtype=float)
    h, w = depth.shape
    if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"Keypoint outside observation: {pixel}")
    col, row = min(int(round(x)), w - 1), min(int(round(y)), h - 1)
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid[row, col]:
        candidates = np.argwhere(valid)
        if not len(candidates):
            raise ValueError("No valid depth in the object mask")
        distances = np.sum((candidates - [y, x]) ** 2, axis=1)
        best = int(np.argmin(distances))
        if distances[best] > max_snap_pixels ** 2:
            raise ValueError(f"No nearby valid depth for {pixel}")
        row, col = candidates[best]
    return get_world_coord_from_pixel([col, row], depth, camera_params)


def ground_truth_receptacle(rigid_states, kind):
    """Recover the generator's target center and yaw from saved shape poses.

    Shape 0 is the floor/bar; shapes 1/2 define the left-to-right axis,
    (cos(yaw), -sin(yaw)) in world x/z. Placement height is supplied separately
    by the action primitive, just as in the demonstration generators.
    """
    if kind not in ("place", "hang"):
        raise ValueError(f"No receptacle for task kind: {kind}")
    shapes, axis, _, _ = rigid_frame(rigid_states)
    return {"kind": kind, "center": [float(shapes[0, 0]), 0.0, float(shapes[0, 2])],
            "angle": float(np.arctan2(-axis[1], axis[0])), "source": "simulator_ground_truth"}
