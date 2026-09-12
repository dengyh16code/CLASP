"""Evaluate semantic perception -> visual planning -> Flex actions on testcases."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

from action_primitives import ActionPrimitives, normalize, parse_plan, split_manipulations
from metric import evaluate
from evaluation_resume import finish_record, resume_case
from simulation_utils import (ROOT, load_cases, task_kind, restore_scene, snapshot, settle,
                              motion_options, pixel_to_world, ground_truth_receptacle,
                              execution_mode, initial_state_type)


class SemanticPlanner:
    """Adapter for the existing semantic_keypoint and request_plan interfaces."""
    def __init__(self, with_vlm=False, prototype_dir=None):
        os.environ.setdefault("CLASP_HOME", str(ROOT))
        from semantic_keypoints.semantic_keypoints import semantic_keypoint
        from utils.gpt_utils import load_prompts
        self.perception = semantic_keypoint(prototype_dir=prototype_dir)
        self.prompts = load_prompts()
        self.with_vlm = with_vlm
        self.prototype_loaded = False
        self.alignment_loaded = False

    def begin_episode(self):
        self.prototype_loaded = False
        self.alignment_loaded = False

    def observe(self, rgb, depth, camera_params, cloth_type, kind, reuse_alignment=False):
        """Refresh image/mask/features; optionally retain episode prototype and yaw."""
        if reuse_alignment and not getattr(self, "alignment_loaded", False):
            raise RuntimeError("Cannot reuse alignment before the initial observation")
        perception = self.perception
        perception.update_observation(rgb)
        if not np.any(perception.obs_mask):
            raise ValueError("Empty cloth segmentation")
        if not self.prototype_loaded:
            perception.load_prototype_image(None)
            self.prototype_loaded = True
        if not reuse_alignment:
            perception.pose_align(with_dino=True, cloth_type=cloth_type)
        pixels = perception.keypoints_extraction(with_vlm=self.with_vlm)
        names = perception.prototype_kp_descriptions
        self.alignment_loaded = True
        world = {name: pixel_to_world(pixel, depth, camera_params, perception.obs_mask)
                 for name, pixel in zip(names, pixels)}
        world.update({f"P{i+1}": world[name] for i, name in enumerate(names)})
        return names, pixels, world

    def plan(self, instruction, names, rgb, pixels, kind):
        from PIL import Image
        from utils.visual_prompt_utils import request_plan, annotate_candidate_keypoints, load_example_pairs
        annotated = annotate_candidate_keypoints(Image.fromarray(rgb), pixels)
        example_images, example_responses = [], []
        for image, data in load_example_pairs(ROOT / "prompts", task=kind):
            example_images.append(image)
            example_responses.append(json.dumps(data))
        return request_plan(instruction, {f"P{i+1}": name for i, name in enumerate(names)},
                            annotated, True, self.prompts, example_images, example_responses)


NON_KEYPOINT_TARGETS = frozenset(("up", "forward", "backward and down",
                                  "forward and down", "box", "rack", "hanger"))


def plan_retry_reason(plan, manipulations, *, step, single_step, available_keypoints=None):
    """Validate task constraints; return None when the parsed plan is acceptable."""
    if plan["completed"]:
        return "Planner marked completed at step 0 before any action" if step == 0 else None
    if single_step and len(manipulations) != 1:
        return f"Single-step task requires exactly one pick-and-release manipulation, got {len(manipulations)}"
    if available_keypoints is not None:
        names = [normalize(name) for name in available_keypoints]
        allowed = set(names) | {f"p{i + 1}" for i in range(len(names))}
        unknown = []
        for group in manipulations:
            for skill, arguments in group:
                contacts = arguments if skill == "grasp" else (
                    [name for name in arguments if name not in NON_KEYPOINT_TARGETS]
                    if skill == "moveto" else [])
                unknown.extend(name for name in contacts if name not in allowed and name not in unknown)
        if unknown:
            choices = ", ".join(names + [f"P{i + 1}" for i in range(len(names))])
            return (f"Plan references unavailable semantic keypoint(s): {', '.join(unknown)}. "
                    f"Use only these current keypoints: {choices}")
    return None


def request_valid_plan(planner, instruction, names, rgb, pixels, kind, *,
                       step, single_step, max_retries, result):
    """Retry on the same observation; retain attempts in the returned result.

    This helper only calls the planner. It never observes or executes actions.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    request = instruction
    for attempt in range(max_retries + 1):
        if attempt:
            result["plan_retries"] += 1
        raw = planner.plan(request, names, rgb, pixels, kind)
        plan = parse_plan(raw)
        result["plans"].append(plan)
        manipulations = [] if plan["completed"] else split_manipulations(plan["subtasks"])

        reason = plan_retry_reason(plan, manipulations, step=step, single_step=single_step,
                                   available_keypoints=names)
        if reason is None:
            return plan, manipulations

        if attempt == max_retries:
            raise ValueError(f"{reason}; exhausted {max_retries} planning retries")
        request = (f"{instruction}\n\n"
                   f"Correction: {reason}. Please revise your plan.\n"
                   f"Previous plan: {json.dumps(plan)}")


def result_directories(case, output):
    """Use the testcase file's episode number as its output folder name."""
    _, episode, state = case["id"].rsplit("/", 2)
    if state != "0":
        raise ValueError("Only one initial state per testcase file is supported")
    test_id = episode
    root = Path(output) / case["task"]
    return root / "success" / test_id, root / "fail" / test_id


def check_output_available(case, output):
    for directory in result_directories(case, output):
        if directory.exists():
            raise FileExistsError(f"Result already exists: {directory}; use another --output root")


def failure_reason(result):
    """Describe execution errors or the specific unsuccessful metric checks."""
    if result.get("error"):
        return result["error"]
    metrics = result.get("metrics", {})
    thresholds = metrics.get("thresholds", {})
    reasons = []
    if not metrics.get("released", True):
        reasons.append("Grippers were not released")
    for value, threshold in (("particle_rmse", "fold_rmse"),
                             ("particle_p95", "fold_p95"),
                             ("support_distance", "hang_support_distance")):
        if value in metrics and metrics[value] > thresholds[threshold]:
            reasons.append(f"{value}={metrics[value]:.6f} exceeds {threshold}={thresholds[threshold]:.6f}")
    for value, threshold in (("coverage", metrics.get("coverage_threshold")),
                             ("inside_fraction", thresholds.get("place_inside_fraction"))):
        if value in metrics and threshold is not None and metrics[value] < threshold:
            reasons.append(f"{value}={metrics[value]:.6f} is below required {threshold:.6f}")
    if "min_height" in metrics and metrics["min_height"] <= thresholds["hang_min_height"]:
        reasons.append(f"min_height={metrics['min_height']:.6f} must exceed "
                       f"{thresholds['hang_min_height']:.6f}")
    if metrics.get("draped") is False:
        reasons.append("Cloth is not draped on both sides of the support bar")
    return "; ".join(reasons) or f"Task unsuccessful (stop reason: {result.get('stop_reason', 'unknown')})"


def run_episode(env, case, planner, output, max_actions=6, max_plan_retries=2):
    """Save the initial and executed final scene; use the goal only for scoring."""
    from PIL import Image
    check_output_available(case, output)
    initial_image = final_image = None
    scene_ready = False
    result = {key: case[key] for key in ("id", "task", "cloth_type", "instruction", "source")}
    result.update(success=False, actions=0, plans=[], plan_retries=0, error=None)
    try:
        kind = task_kind(case["task"])
        mode = execution_mode(case)
        start_type = initial_state_type(case)
        fixed_plan = start_type == "flat"
        result["initial_state_type"] = start_type
        result["single_step"] = case["single_step"]
        result["flat"] = case["flat"]
        result["planning_mode"] = "fixed_plan" if fixed_plan else "replanning"
        result["execution_mode"] = mode
        receptacle = ground_truth_receptacle(case["rigid_states"], kind) if kind in ("place", "hang") else None
        result["receptacle_source"] = "simulator_ground_truth" if receptacle is not None else None
        restore_scene(env, case["config"], case["state"], case["task"], case["rigid_states"])
        scene_ready = True
        initial_image = Image.fromarray(env.render_image()[0]).copy()
        executor = ActionPrimitives(env)
        planner.begin_episode()
        pending = []
        for step in range(max_actions):
            rgb, depth = env.render_image()
            names, pixels, world = planner.observe(rgb, depth, env.camera_params,
                                                  case["cloth_type"], kind, reuse_alignment=step > 0)
            if step == 0 or not fixed_plan:
                plan, pending = request_valid_plan(
                    planner, case["instruction"], names, rgb, pixels, kind,
                    step=step, single_step=fixed_plan and mode == "single_step",
                    max_retries=max_plan_retries, result=result)
                if plan["completed"]:
                    result["stop_reason"] = "planner_completed"
                    break
            group = pending.pop(0)
            # Ground only the next manipulation, using this observation.
            action = executor.compile(group, world, receptacle)
            executor.execute(action)
            result["actions"] += 1
            settle(env)
            if fixed_plan and not pending:
                result["stop_reason"] = "single_step_executed" if mode == "single_step" else "plan_executed"
                break
        else:
            result["stop_reason"] = "action_budget"
        env.set_grasp(False)
        env.movep(env.reset_pos, speed=0.5, min_steps=1)
        settled = settle(env)
        final = snapshot(env)
        released = not any(env.grasp_states) and all(p is None for p in env.action_tool.picked_particles)
        result["metrics"] = evaluate(case["task"], final, case["goal_state"], case["cloth_type"],
                                     case["rigid_states"], released=released,
                                     particle_radius=env.particle_radius,
                                     single_step=case["single_step"])
        result["settled"] = settled
        # Settling is diagnostic only, not an additional success requirement.
        result["success"] = bool(result["metrics"]["success"])
        final_image = Image.fromarray(env.render_image()[0]).copy()
    except Exception as error:
        result["success"] = False
        result["stop_reason"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
        # A failed API call or action still needs an image of the actual final scene.
        if scene_ready:
            try:
                final_image = Image.fromarray(env.render_image()[0]).copy()
            except Exception as render_error:
                result["error"] += f"; final image unavailable: {render_error}"
        if initial_image is None:
            result["error"] += "; initial image unavailable"
        if not scene_ready:
            result["error"] += "; final image unavailable (scene initialization failed)"
    directory = result_directories(case, output)[0 if result["success"] else 1]
    directory.mkdir(parents=True, exist_ok=False)
    (directory / ".in-progress").touch()
    if initial_image is not None:
        initial_image.save(directory / "initial.png")
    if final_image is not None:
        final_image.save(directory / "final.png")
    if not result["success"]:
        result["failure_reason"] = failure_reason(result)
        (directory / "failure_reason.txt").write_text(result["failure_reason"] + "\n")
    finish_record(directory, result)
    return result


def summarize(results):
    def stats(items):
        return {"cases": len(items), "successes": sum(bool(r["success"]) for r in items),
                "success_rate": sum(bool(r["success"]) for r in items) / len(items) if items else 0.0,
                "errors": sum(r.get("error") is not None for r in items)}
    summary = stats(results)
    summary["by_task"] = {task: stats([result for result in results if result["task"] == task])
                          for task in sorted({result["task"] for result in results})}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-cases", default="test_case")
    parser.add_argument("--output", default="eval_results",
                        help="Result root: <root>/<task>/success|fail/<test_id> (default: eval_results)")
    parser.add_argument("--task")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed success/fail cases; discard partial results and rerun them")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-actions", type=int, default=6,
                        help="Action limit for non-flatten tasks (default: 6)")
    parser.add_argument("--max-flatten-actions", type=int, default=3,
                        help="Action limit for flatten tasks, independent of --max-actions (default: 3)")
    parser.add_argument("--max-plan-retries", type=int, default=2,
                        help="Extra VLM calls for step-0 completion or invalid flat single-step action counts")
    parser.add_argument("--render-dim", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prototype-dir", default=str(ROOT / "semantic_keypoints" / "prototype_simu"),
                        help="Prototype image/mask/JSON folder; defaults to simulation prototypes")
    parser.add_argument("--with-vlm", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Inspect testcase schema without Flex or model calls")
    args = parser.parse_args()
    if args.max_actions <= 0 or args.render_dim <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("max-actions, render-dim, and limit must be positive")
    if args.max_flatten_actions <= 0:
        parser.error("max-flatten-actions must be positive")
    if args.max_plan_retries < 0:
        parser.error("max-plan-retries must be nonnegative")
    cases = [case for case in load_cases(args.test_cases) if not args.task or case["task"] == args.task]
    if args.limit is not None and (not args.resume or args.validate_only):
        cases = cases[:args.limit]
    if not cases:
        parser.error("No matching testcases")
    if len({case["id"] for case in cases}) != len(cases):
        parser.error("Duplicate testcase IDs")
    if args.validate_only:
        print(json.dumps({"cases": len(cases), "tasks": sorted({c["task"] for c in cases}),
                          "initial_state_types": sorted({initial_state_type(c) for c in cases})}, indent=2))
        return
    output = Path(args.output)
    results = []
    if args.resume:
        pending = []
        for case in cases:
            try:
                completed = resume_case(case, output, result_directories(case, output))
            except ValueError as error:
                parser.error(str(error))
            if completed is None:
                pending.append(case)
            else:
                results.append(completed)
                print(f"Skipping evaluated case: {case['id']} success={completed['success']}", flush=True)
        cases = pending if args.limit is None else pending[:args.limit]
        print(f"Resume: {len(results)} already evaluated; {len(cases)} cases to run "
              f"({len(pending)} unfinished total)", flush=True)
        if not cases:
            print(json.dumps(summarize(results), indent=2))
            return
    try:
        for case in cases:
            check_output_available(case, output)
    except FileExistsError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    print(f"Writing evaluation results to {output}", flush=True)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    import pyflex
    from softgym.envs.cloth_env import ClothEnv
    planner = SemanticPlanner(args.with_vlm, args.prototype_dir)
    env = ClothEnv(gui=False, cloth3d=bool(cases[0]["cloth3d"]), render_dim=args.render_dim)
    try:
        for case in cases:
            env.cloth3d = bool(case["cloth3d"])
            for name, value in motion_options(case["task"], case["cloth_type"]).items():
                setattr(env, name, value)
            action_limit = args.max_flatten_actions if task_kind(case["task"]) == "flatten" else args.max_actions
            result = run_episode(env, case, planner, output, action_limit, args.max_plan_retries)
            results.append(result)
            print(f"{case['id']}: success={result['success']} actions={result['actions']} error={result['error']}", flush=True)
            if not result["success"]:
                print(result["failure_reason"], flush=True)
    finally:
        pyflex.clean()
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
