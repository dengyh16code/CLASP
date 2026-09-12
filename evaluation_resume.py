"""Completion records and interrupted-output handling for evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finish_record(directory, result):
    """Commit last, after all available images and failure details are saved."""
    directory = Path(directory)
    artifacts = {name: digest(directory / name)
                 for name in ("initial.png", "final.png", "failure_reason.txt")
                 if (directory / name).is_file()}
    record = {"schema_version": 1, "result": result, "artifacts": artifacts}
    temporary = directory / ".result.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / "result.json")
    (directory / ".in-progress").unlink(missing_ok=True)


def completed_result(directory, case, success):
    """Validate a new completion record, or recognize fully saved legacy output."""
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError(f"Refusing to resume a symlinked result directory: {directory}")
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise ValueError(f"Result path is not a directory: {directory}")
    marker = directory / "result.json"
    if marker.exists():
        try:
            record = json.loads(marker.read_text())
            result, artifacts = record["result"], record["artifacts"]
            if (record["schema_version"] != 1 or result["id"] != case["id"] or
                    result["task"] != case["task"] or result["success"] is not success or
                    not isinstance(artifacts, dict)):
                return None
            required = {"initial.png", "final.png"} if success or not result.get("error") else set()
            if not success:
                required.add("failure_reason.txt")
            if not required <= artifacts.keys():
                return None
            if set(artifacts) - {"initial.png", "final.png", "failure_reason.txt"}:
                return None
            if any(digest(directory / name) != expected for name, expected in artifacts.items()):
                return None
            return result
        except (OSError, ValueError, KeyError, TypeError):
            return None
    # Never mistake an interrupted new-format write for a complete legacy case.
    if (directory / ".in-progress").exists() or (directory / ".result.json.tmp").exists():
        return None
    from PIL import Image
    try:
        for name in ("initial.png", "final.png"):
            with Image.open(directory / name) as image:
                image.verify()
        reason = None if success else (directory / "failure_reason.txt").read_text().strip()
        if not success and not reason:
            return None
    except (OSError, ValueError, SyntaxError):
        return None
    # Legacy files lack detailed metrics/plans; do not invent those values.
    result = {key: case[key] for key in (
        "id", "task", "cloth_type", "single_step", "flat", "initial_state_type"
    ) if key in case}
    result.update(success=success, completion_source="legacy_artifacts")
    if reason:
        result["failure_reason"] = reason
    return result


def resume_case(case, output, directories):
    """Return a completed result; discard incomplete directories before a rerun."""
    directories = tuple(directories)
    for directory in directories:
        relative = directory.relative_to(output)
        if (len(relative.parts) != 3 or relative.parts[1] not in ("success", "fail") or
                any(part in (".", "..") for part in relative.parts) or
                directory.parent.is_symlink() or directory.parent.parent.is_symlink()):
            raise ValueError(f"Unexpected or symlinked result directory: {directory}")
    records = [completed_result(directory, case, success)
               for directory, success in zip(directories, (True, False))]
    if sum(record is not None for record in records) > 1:
        raise ValueError(f"Both success and fail contain completed results for {case['id']}")
    for directory, record in zip(directories, records):
        if record is None and directory.exists():
            # Only this case's exact success/fail leaf can be removed.
            shutil.rmtree(directory)
            print(f"Removed incomplete result: {directory}; case will be rerun if selected", flush=True)
    return next((record for record in records if record is not None), None)
