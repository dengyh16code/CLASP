"""Ground the planner's grasp/moveto/pull/rotate/release language in ClothEnv.

Each released manipulation is compiled to the existing demonstration
single/dual pick-place, fling, or rigid-object placement trajectory. No eval or
arbitrary code execution is used. Coordinates are world (x, y, z), in metres.
"""
import json
import re
import numpy as np


def normalize(name):
    return " ".join(name.lower().replace("_", " ").replace("waitband", "waistband").split())


def parse_plan(response):
    if isinstance(response, str):
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        response = json.loads(text)
    if not isinstance(response, dict) or type(response.get("completed")) is not bool:
        raise ValueError("Plan must contain a boolean completed field")
    subtasks = response.get("subtasks")
    if not isinstance(subtasks, list) or not all(isinstance(s, str) for s in subtasks):
        raise ValueError("Plan subtasks must be a list of skill strings")
    if response["completed"] and subtasks:
        raise ValueError("A completed plan cannot contain actions")
    if not response["completed"] and not subtasks:
        raise ValueError("An incomplete plan must contain actions")
    return response


def split_manipulations(subtasks):
    groups, group = [], []
    for subtask in subtasks:
        match = re.fullmatch(r"\s*(grasp|moveto|release|rotate|pull)\s*\(([^()]*)\)\s*", subtask, re.I)
        if not match:
            raise ValueError(f"Invalid skill: {subtask!r}")
        name, text = match.groups()
        args = [normalize(s) for s in text.split(",")] if text.strip() else []
        name = name.lower()
        if name == "grasp" and group or not group and name != "grasp":
            raise ValueError("Each manipulation must start with grasp and end with release")
        if name in ("release", "pull", "rotate") and args:
            raise ValueError(f"{name} takes no arguments")
        group.append((name, args))
        if name == "release":
            groups.append(group)
            group = []
    if group:
        raise ValueError("Plan leaves a gripper holding cloth")
    return groups


class ActionPrimitives:
    def __init__(self, env):
        self.env = env

    def compile(self, group, keypoints, receptacle=None):
        points = {normalize(k): np.asarray(v, dtype=float) for k, v in keypoints.items()}
        def resolve(names):
            if len(names) not in (1, 2):
                raise ValueError("Expected one or two contact points")
            values = []
            for name in names:
                if name not in points:
                    raise ValueError(f"Unknown semantic keypoint: {name}")
                value = points[name]
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid world coordinate for {name}")
                values.append(value.copy())
            return values
        picks = resolve(group[0][1])
        middle = group[1:-1]
        if len(middle) == 1 and middle[0][0] == "moveto":
            places = resolve(middle[0][1])
            if len(picks) != len(places):
                raise ValueError("Pick and place must use the same number of arms")
            return {"primitive": "single" if len(picks) == 1 else "dual", "picks": picks, "places": places}
        if len(picks) != 2:
            raise ValueError("Flatten, hang, and place require two grasp points")
        if middle == [("moveto", ["up"]), ("pull", []), ("moveto", ["forward"]),
                      ("moveto", ["backward and down"])]:
            return {"primitive": "flatten", "picks": picks}
        allowed = [("moveto", ["up"])]
        if len(middle) > 1 and middle[1] == ("rotate", []):
            allowed.append(("rotate", []))
        targets = [args[0] for name, args in middle if name == "moveto" and len(args) == 1
                   and args[0] in ("box", "rack", "hanger")]
        if len(targets) == 1:
            target = targets[0]
            allowed += [("moveto", [target]), ("moveto", ["forward and down"])]
            kind = "place" if target == "box" else "hang"
            if middle == allowed and receptacle is not None and receptacle["kind"] == kind:
                return {"primitive": kind, "picks": picks, "receptacle": receptacle}
        raise ValueError(f"Unsupported manipulation sequence or missing receptacle: {group}")

    def execute(self, action):
        env, kind = self.env, action["primitive"]
        picks = [np.asarray(p, dtype=float).copy() for p in action["picks"]]
        try:
            if kind == "single":
                env.pick_and_place_single(picks[0], action["places"][0].copy())
            elif kind == "dual":
                env.pick_and_place_dual(picks[0], action["places"][0].copy(), picks[1], action["places"][1].copy())
            elif kind == "flatten":
                if env.pick_and_fling(*picks) is False:
                    raise RuntimeError("Fling failed to grasp cloth")
            elif kind in ("place", "hang"):
                obj = action["receptacle"]
                env.place_on_rigid_obj(*picks, np.asarray(obj["center"]).copy(),
                                       obj["angle"], 0.10 if kind == "place" else 0.15,
                                       0.8 if kind == "place" else 0.4)
            else:
                raise ValueError(f"Unsupported primitive: {kind}")
        finally:
            # set_grasp only updates flags; movep applies them to particle masses.
            env.set_grasp(False)
            env.movep(env.reset_pos, speed=0.5, min_steps=1)

    def execute_plan(self, response, keypoints, receptacle=None, first_only=False):
        plan = parse_plan(response)
        groups = split_manipulations(plan["subtasks"])
        actions = [self.compile(group, keypoints, receptacle) for group in groups]
        for action in actions[:1] if first_only else actions:
            self.execute(action)
        return actions[:1] if first_only else actions
