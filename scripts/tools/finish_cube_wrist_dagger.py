"""Wait on a training process exit event, then evaluate its completed checkpoint."""

import argparse
import ctypes
import errno
import json
import os
import select
import subprocess
import sys
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pid", type=int, required=True)
  parser.add_argument("--training", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.training = args.training.resolve()
  args.output = args.output.resolve()
  root = Path(__file__).resolve().parents[2]
  status = args.training / "pipeline_status.json"

  def save(stage, **fields):
    status.write_text(json.dumps({"stage": stage, **fields}, indent=2))

  save("waiting_for_training", training_pid=args.pid)
  try:
    if hasattr(os, "pidfd_open"):
      fd = os.pidfd_open(args.pid)
    else:
      libc = ctypes.CDLL(None, use_errno=True)
      fd = libc.pidfd_open(args.pid, 0)
      if fd < 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
          raise ProcessLookupError(error, os.strerror(error))
        raise OSError(error, os.strerror(error))
  except ProcessLookupError:
    fd = None
  if fd is not None:
    # Kernel process-exit notification, without repeated process polling.
    select.select([fd], [], [])
    os.close(fd)
  report_path = args.training / "training.json"
  if (
    not report_path.exists() or len(json.loads(report_path.read_text())["rounds"]) != 3
  ):
    save(
      "training_incomplete",
      reason="Expected three completed training rounds; see training log",
    )
    sys.exit(1)
  save("evaluating", checkpoint=str(args.training / "student.pt"))
  with (args.training / "evaluation.log").open("w") as log:
    result = subprocess.run(
      [
        str(root / ".venv/bin/python"),
        str(root / "scripts/eval_cube_wrist_student.py"),
        "--student",
        str(args.training / "student.pt"),
        "--output",
        str(args.output),
        "--trials",
        "16",
        "--seconds",
        "32",
        "--seed",
        "7400",
      ],
      cwd=root,
      stdout=log,
      stderr=subprocess.STDOUT,
    )
  if result.returncode:
    save("evaluation_failed", returncode=result.returncode)
    sys.exit(result.returncode)
  baseline = json.loads(
    (
      root
      / "hardware/franka_wuji_sim_assembly_20260930/visual_closed_loop_backoff40/evaluation.json"
    ).read_text()
  )
  measured = json.loads((args.output / "evaluation.json").read_text())
  fields = [
    "goal_successes",
    "goal_attempts",
    "goal_success_rate",
    "survived_trials",
    "survival_rate",
  ]
  comparison = {
    "baseline": {k: baseline[k] for k in fields},
    "camera_dagger": {k: measured[k] for k in fields},
    "seed": 7400,
    "trials": 16,
    "seconds": 32,
    "success_rate_change_percentage_points": 100
    * (measured["goal_success_rate"] - baseline["goal_success_rate"]),
    "scope": "Matched simulation benchmark; no real hardware result or unseen-seed generalization claim",
  }
  (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2))
  save("complete", evaluation=str(args.output), **comparison)


if __name__ == "__main__":
  main()
