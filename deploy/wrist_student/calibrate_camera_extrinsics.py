"""Measure the real wrist-camera pose relative to the hand from recorded runs.

The black-gloved hand is segmented on the light background in cube-free frames (before the
cube is first detected and within --max-time of the run start), the hand MJCF is posed with
the logged joint angles, and a 6-DoF correction of the release camera extrinsics is fitted
so the projected hand mesh points fall on the glove mask (and the mask is covered by the
points). Writes camera_extrinsics_measured.json (camera_to_tag rotation/origin in the same
convention as policy_config.json) plus before/after overlays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_handover import TAG_IN_PALM_POS  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "camera_extrinsics_measured.json"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("runs", type=Path, nargs="+")
  p.add_argument("--max-time", type=float, default=6.0, help="Only frames this early in a run (cube not yet near)")
  p.add_argument("--frames-per-run", type=int, default=12)
  p.add_argument("--dark", type=int, default=75, help="Gray threshold for the black glove")
  p.add_argument("--output", type=Path, default=OUTPUT)
  a = p.parse_args()
  a.output.parent.mkdir(parents=True, exist_ok=True)

  cfg = json.loads((ROOT / "release_models/policy_config.json").read_text())
  calib = json.loads(
    (ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json").read_text()
  )
  K = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1.0]])
  dist = np.array(calib["coeffs"])
  Rcam = np.array(cfg["camera_to_tag_rotation"])
  pcam = np.array(cfg["camera_origin_tag_m"])
  T_tag = np.array(cfg["tag_in_palm_rotation"])
  R_pc = Rcam.T @ T_tag.T  # palm -> optical camera
  t_pc = -Rcam.T @ (T_tag.T @ TAG_IN_PALM_POS + pcam)

  m = mujoco.MjModel.from_xml_path(str(ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_mjlab.xml"))
  dm = mujoco.MjData(m)
  qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in cfg["joint_names"]]
  mesh_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 1]

  def hand_points(q):
    dm.qpos[qadr] = q
    mujoco.mj_kinematics(dm, dm) if False else mujoco.mj_kinematics(m, dm)
    pts = []
    for g in mesh_geoms:
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
      v = v[:: max(1, len(v) // 150)]
      pts.append(dm.geom_xpos[g] + v @ dm.geom_xmat[g].reshape(3, 3).T)
    return np.concatenate(pts)

  frames = []
  for run in a.runs:
    d = np.load(run / "trajectory.npz")
    t = d["t"] - d["t"][0]
    first_seen = int(np.argmax(d["usable"])) if d["usable"].any() else len(t)
    cand = [i for i in range(first_seen) if t[i] <= a.max_time]
    if not cand or "frame" not in d.files:
      print(f"{run.name}: no usable cube-free frames")
      continue
    video = cv2.VideoCapture(str(run / "wrist.mp4"))
    for i in np.array(cand)[np.linspace(0, len(cand) - 1, min(a.frames_per_run, len(cand))).astype(int)]:
      video.set(cv2.CAP_PROP_POS_FRAMES, int(d["frame"][i]))
      ok, img = video.read()
      if not ok:
        continue
      mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < a.dark).astype(np.uint8)
      mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
      mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
      dt = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
      edge = np.argwhere(cv2.Canny(mask * 255, 50, 150) > 0)[:, ::-1].astype(float)
      edge = edge[(edge[:, 1] > 8)]  # the palm leaves the image at the top border
      frames.append({"img": img, "dt": np.minimum(dt, 60), "edge": edge[:: max(1, len(edge) // 600)],
                     "pts": hand_points(d["q"][i]), "name": f"{run.name}@{t[i]:.1f}s"})
  if len(frames) < 3:
    raise SystemExit("Too few cube-free frames")
  print(f"{len(frames)} frames")

  def project(x, pts):
    E = Rotation.from_rotvec(x[:3]).as_matrix()
    cam = (pts @ R_pc.T + t_pc) @ E.T + x[3:]
    ok = cam[:, 2] > 0.01
    uv = cv2.projectPoints(cam[ok], np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    return uv

  def cost(x, detail=False):
    total = []
    for f in frames:
      uv = project(x, f["pts"])
      inside = (uv[:, 0] >= 0) & (uv[:, 0] < 639) & (uv[:, 1] >= 0) & (uv[:, 1] < 479)
      model_to_mask = f["dt"][uv[inside, 1].astype(int), uv[inside, 0].astype(int)].mean() if inside.any() else 60
      mask_to_model = np.minimum(cKDTree(uv[inside]).query(f["edge"])[0], 60).mean() if inside.sum() > 10 else 60
      total.append(model_to_mask + 0.5 * mask_to_model)
    return total if detail else float(np.mean(total))

  x0 = np.zeros(6)
  best = None
  for start in ([0, 0, 0, 0, 0, 0], [0, 0.15, 0, 0, 0, 0], [0, -0.15, 0, 0, 0, 0]):
    r = minimize(cost, np.array(start, float), method="Powell",
                 bounds=[(-0.35, 0.35)] * 3 + [(-0.04, 0.04)] * 3, options={"xtol": 1e-3, "ftol": 1e-3})
    print(f"start {start}: cost {r.fun:.2f} px, x {np.round(r.x, 4).tolist()}", flush=True)
    if best is None or r.fun < best.fun:
      best = r
  x = best.x
  before, after = cost(x0, True), cost(x, True)
  E = Rotation.from_rotvec(x[:3]).as_matrix()
  R_pc_new, t_pc_new = E @ R_pc, E @ t_pc + x[3:]
  Rcam_new = T_tag.T @ R_pc_new.T
  pcam_new = -Rcam_new @ (R_pc_new @ TAG_IN_PALM_POS + t_pc_new)
  cam_in_palm_old = -R_pc.T @ t_pc
  cam_in_palm_new = -R_pc_new.T @ t_pc_new
  result = {
    "camera_to_tag_rotation": Rcam_new.tolist(),
    "camera_origin_tag_m": pcam_new.tolist(),
    "correction_rotvec_camera_rad": x[:3].tolist(),
    "correction_rotation_deg": float(np.degrees(np.linalg.norm(x[:3]))),
    "camera_shift_in_palm_mm": ((cam_in_palm_new - cam_in_palm_old) * 1000).tolist(),
    "cost_px_before": float(np.mean(before)),
    "cost_px_after": float(np.mean(after)),
    "frames": [f["name"] for f in frames],
    "note": "Fitted from glove silhouette vs hand MJCF at logged joint angles (calibrate_camera_extrinsics.py).",
  }
  a.output.write_text(json.dumps(result, indent=2))
  print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "camera_to_tag_rotation")}, indent=2))
  out = a.output.parent / "camera_extrinsics_fit"
  out.mkdir(exist_ok=True)
  for k, f in enumerate(frames[:: max(1, len(frames) // 4)]):
    tiles = []
    for label, xx in (("release", x0), ("fitted", x)):
      img = f["img"].copy()
      for u, v in project(xx, f["pts"]):
        if 0 <= u < 640 and 0 <= v < 480:
          cv2.circle(img, (int(u), int(v)), 1, (0, 255, 0) if label == "fitted" else (0, 0, 255), -1)
      cv2.putText(img, f"{label} {f['name']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
      tiles.append(img)
    cv2.imwrite(str(out / f"fit_{k}.jpg"), np.hstack(tiles))
  print(f"Wrote {a.output} and overlays in {out}")


if __name__ == "__main__":
  main()
