"""Compute the pre-grasp 'open' pose: the training grasp widened just enough for the cube
to be pushed up into the hand from below (palm down) without touching the fingers.

Kinematics only (no dynamics) on the trained hand MJCF. Distances come from the regular
collision pipeline (contacts within a 30 mm cube margin): mj_geomDistance returned spurious
zeros for the thumb's soft-body mesh geom. The cube
moves along the palm normal from 80 mm below its sim reset position up to it, for cube
yaws 0 and 45 deg about the palm normal. Fingers must also keep SELF_MARGIN from each
other at the open pose and along the closing blend (the first version spread index and
middle into each other on the real hand). Coordinate ascent over joint offsets from
default_joint_pos, inside the soft joint limits, maximising summed clearance below the
margin with a small pull toward the training grasp (short closing motion).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "pregrasp_open_pose.json"
SENSE_M = 0.03
SELF_MARGIN_M = 0.003


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--margin-mm", type=float, default=8.0)
  p.add_argument("--approach-mm", type=float, default=80.0)
  p.add_argument("--output", type=Path, default=OUTPUT)
  a = p.parse_args()

  cfg = json.loads((ROOT / "release_models/policy_config.json").read_text())
  q_default = np.array(cfg["default_joint_pos"])
  limits = np.array(cfg["soft_joint_pos_limits"])
  names = cfg["joint_names"]
  half = cfg["cube_edge_m"] / 2
  spec = mujoco.MjSpec.from_file(str(ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_mjlab.xml"))
  body = spec.worldbody.add_body(name="cube", mocap=True)
  body.add_geom(
    name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half] * 3, contype=0xFFFF, conaffinity=0xFFFF, margin=SENSE_M
  )
  m = spec.compile()
  d = mujoco.MjData(m)
  cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube")
  body_name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
  fingers = [
    g for g in range(m.ngeom)
    if g != cube and (m.geom_contype[g] or m.geom_conaffinity[g]) and "palm" not in body_name(g)
  ]
  finger_of = [body_name(g).split("_")[1] for g in fingers]
  # Finger-finger proximity is read from contacts generated within this geom margin; the
  # cube's larger margin still governs cube contacts (MuJoCo uses the max of the pair).
  for g in fingers:
    m.geom_margin[g] = 0.01
  finger_by_geom = dict(zip(fingers, finger_of))
  qpos_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
  # Sim reset cube position in the palm frame (scripts/train/train_cube_downward.py:make_cfg).
  nominal = np.array([half + 0.003, 0.009, 0.105])
  path = [nominal + np.array([s, 0, 0]) for s in np.linspace(a.approach_mm / 1000, 0, 17)]
  yaws = (0.0, 45.0)

  column = {g: i for i, g in enumerate(fingers)}

  def distances(q, yaw):
    """(path, finger geoms) signed distances, SENSE_M where no contact within the margin."""
    d.qpos[:] = 0
    d.qpos[qpos_adr] = q
    d.mocap_quat[0] = Rotation.from_euler("x", yaw, degrees=True).as_quat()[[3, 0, 1, 2]]
    rows = np.full((len(path), len(fingers)), SENSE_M)
    for k, pos in enumerate(path):
      d.mocap_pos[0] = pos
      mujoco.mj_forward(m, d)
      for c in d.contact[: d.ncon]:
        other = c.geom2 if c.geom1 == cube else c.geom1 if c.geom2 == cube else None
        if other in column:
          rows[k, column[other]] = min(rows[k, column[other]], c.dist)
    return rows

  margin = a.margin_mm / 1000

  def self_distances(q):
    """Signed distances between geoms of different fingers (cube moved far away)."""
    d.qpos[:] = 0
    d.qpos[qpos_adr] = q
    d.mocap_pos[0] = [1.0, 1.0, 1.0]
    mujoco.mj_forward(m, d)
    out = [0.01]
    for c in d.contact[: d.ncon]:
      fa, fb = finger_by_geom.get(c.geom1), finger_by_geom.get(c.geom2)
      if fa and fb and fa != fb:
        out.append(c.dist)
    return np.array(out)

  def self_shortfall(q):
    poses = [q + k * (q_default - q) for k in (0.0, 0.25, 0.5, 0.75)]
    return sum(np.minimum(self_distances(x) - SELF_MARGIN_M, 0).sum() for x in poses)

  def objective(q):
    shortfall = sum(np.minimum(distances(q, y) - margin, 0).sum() for y in yaws)
    return shortfall + 10 * self_shortfall(q) - 0.0005 * np.abs(q - q_default).sum()

  q = np.clip(q_default, limits[:, 0], limits[:, 1])
  best = objective(q)
  for step in (0.4, 0.2, 0.1, 0.05):
    improved = True
    while improved:
      improved = False
      for i in range(len(q)):
        for sign in (1, -1):
          cand = q.copy()
          cand[i] = np.clip(cand[i] + sign * step, limits[i, 0], limits[i, 1])
          value = objective(cand)
          if value > best + 1e-9:
            q, best, improved = cand, value, True

  clearance = {}
  for yaw in yaws:
    D = distances(q, yaw)
    clearance[f"yaw_{yaw:.0f}"] = {
      f: round(float(D[:, [i for i, x in enumerate(finger_of) if x == f]].min() * 1000), 1)
      for f in sorted(set(finger_of))
    }
  self_min = {
    f"blend_{k:.2f}": round(float(self_distances(q + k * (q_default - q)).min() * 1000), 1)
    for k in (0.0, 0.25, 0.5, 0.75, 1.0)
  }
  result = {
    "joint_names": names,
    "finger_finger_min_mm_open_to_default": self_min,
    "q_open": [round(float(v), 4) for v in q],
    "delta_from_default": [round(float(v), 4) for v in q - q_default],
    "min_clearance_mm_along_insertion": clearance,
    "margin_mm": a.margin_mm,
    "approach_mm": a.approach_mm,
    "note": "Kinematic clearance only; the tightest spots are at the final (palm-contact) position.",
  }
  a.output.write_text(json.dumps(result, indent=2))
  print(json.dumps({k: result[k] for k in ("min_clearance_mm_along_insertion", "finger_finger_min_mm_open_to_default", "margin_mm")}, indent=2))
  print(f"max |delta| {np.abs(q - q_default).max():.2f} rad; wrote {a.output}")


if __name__ == "__main__":
  main()
