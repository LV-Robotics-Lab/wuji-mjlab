"""Fit how each printed sticker sheet actually sits on the cube, from captured frames.

Each face's sticker is internally rigid, but a sheet may be rotated in-plane or
sit on a different face than cube_tags45.json says. Because the cube is symmetric,
re-labelling is harmless as long as the board is rigidly consistent, so each face
gets one of the 24 cube rotations M_f (model face coords -> physical coords),
measured relative to a reference face. Marker edge and marker offset are then fitted
relative to the cube size (images fix only ratios; --cube-edge-mm is the absolute
anchor, so measure it). Input: dirs written by check_cube_tags_live.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_handover import build_board  # noqa: E402

DICTS = {"aruco4x4_50": cv2.aruco.DICT_4X4_50, "tag36h11": cv2.aruco.DICT_APRILTAG_36h11}
CUBE_GROUP = [
  m
  for p in itertools.permutations(range(3))
  for s in itertools.product((1, -1), repeat=3)
  if np.isclose(np.linalg.det(m := np.eye(3)[list(p)] * np.array(s)[:, None]), 1)
]


def angle(R):
  return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def snap(R, tol_deg):
  best = min(CUBE_GROUP, key=lambda G: angle(G.T @ R))
  return best if angle(best.T @ R) < tol_deg else None


def key(M):
  return tuple(np.rint(M).astype(int).ravel())


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("captures", type=Path, nargs="+")
  p.add_argument("--family", default="aruco4x4_50", choices=list(DICTS))
  p.add_argument("--marker-edge-mm", type=float, default=10.8333)
  p.add_argument("--cube-edge-mm", type=float, default=45.0, help="Measured cube edge (absolute scale anchor)")
  p.add_argument("--reference", default="TOP")
  p.add_argument("--output", type=Path, default=ROOT / "hardware/palm_down_cube_45mm/board_measured_aruco4x4.json")
  a = p.parse_args()

  cube = json.loads((ROOT / "hardware/palm_down_cube_45mm/cube_tags45.json").read_text())
  offset = 100 if a.family == "tag36h11" else 0
  face_of = {int(i) + offset: f for f, lay in cube["faces_config"].items() for i in lay}
  model = build_board(ROOT / "hardware/palm_down_cube_45mm/cube_tags45.json", a.family, a.marker_edge_mm / 1000)
  obj = {f["id"]: np.array(f["corners_object_m"]) for f in model["faces"]}
  calib = json.loads(
    (ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json").read_text()
  )
  K = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1.0]])
  dist = np.array(calib["coeffs"])
  params = cv2.aruco.DetectorParameters()
  params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
  detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTS[a.family]), params)

  def solve(points_obj, points_img):
    best = None
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_IPPE):
      ok, rvs, tvs, _ = cv2.solvePnPGeneric(points_obj, points_img, K, dist, flags=flag)
      for rv, tv in zip(rvs, tvs) if ok else []:
        err = np.linalg.norm(cv2.projectPoints(points_obj, rv, tv, K, dist)[0].reshape(-1, 2) - points_img, axis=1)
        if tv[2, 0] > 0 and (best is None or err.mean() < best[2]):
          best = (cv2.Rodrigues(rv)[0], tv.ravel(), float(err.mean()))
    return best

  frames = []
  for capture in a.captures:
    for png in sorted(capture.glob("frame*_raw.png")):
      corners, ids, _ = detector.detectMarkers(cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2GRAY))
      if ids is None:
        continue
      groups = defaultdict(list)
      for i, c in zip(ids.ravel().tolist(), corners):
        if i in face_of:
          groups[face_of[i]].append((i, c[0].astype(np.float64)))
      poses = {}
      for face, ms in groups.items():
        if len(ms) >= 2:
          s = solve(np.concatenate([obj[i] for i, _ in ms]), np.concatenate([c for _, c in ms]))
          if s is not None and s[2] < 2.0:
            poses[face] = s
      frames.append((png, groups, poses))

  # Pairwise relative placements S_ab = M_a^T M_b, snapped to the cube group.
  votes = defaultdict(Counter)
  for _, _, poses in frames:
    for fa, fb in itertools.permutations(poses, 2):
      S = snap(poses[fa][0].T @ poses[fb][0], 10.0)
      if S is not None:
        votes[(fa, fb)][key(S)] += 1
  placement = {a.reference: np.eye(3)}
  changed = True
  while changed:
    changed = False
    for (fa, fb), counter in votes.items():
      if fa in placement and fb not in placement:
        S = np.array(counter.most_common(1)[0][0], float).reshape(3, 3)
        placement[fb] = placement[fa] @ S
        changed = True

  # Fit marker edge / offset with the cube half-size anchored, using every frame with
  # >= 2 markers on placed faces (joint PnP across faces).
  half = a.cube_edge_mm / 2000 + 0.00001
  cube_json = ROOT / "hardware/palm_down_cube_45mm/cube_tags45.json"

  def make_table(edge_m, offset_m):
    nominal = build_board(cube_json, a.family, edge_m, offset_m, half)
    return {
      f["id"]: np.array(f["corners_object_m"]) @ placement[face_of[f["id"]]].T
      for f in nominal["faces"]
      if face_of[f["id"]] in placement
    }

  def mean_error(table):
    errs = []
    for _, groups, _ in frames:
      ms = [(i, c) for g in groups.values() for i, c in g if i in table]
      if len(ms) >= 2:
        s = solve(np.concatenate([table[i] for i, _ in ms]), np.concatenate([c for _, c in ms]))
        if s is not None:
          errs.append(s[2])
    return float(np.mean(errs))

  start_edge, start_offset = a.marker_edge_mm / 1000, cube["tag_center_offset"]
  fit = minimize(
    lambda x: mean_error(make_table(start_edge * x[0], start_offset * x[1])),
    [1.0, 1.0],
    method="Nelder-Mead",
    options={"xatol": 1e-4, "fatol": 1e-4},
  )
  edge_m, offset_m = start_edge * fit.x[0], start_offset * fit.x[1]
  obj = {
    f["id"]: np.array(f["corners_object_m"]) for f in build_board(cube_json, a.family, edge_m, offset_m, half)["faces"]
  }

  # Faces only ever seen as single markers next to a placed face: test all 24 placements.
  single = defaultdict(Counter)
  fitted = make_table(edge_m, offset_m)
  for _, groups, poses in frames:
    if not any(f in placement for f in poses):
      continue
    ms_placed = [(i, c) for f, g in groups.items() if f in placement for i, c in g]
    reference_pose = solve(np.concatenate([fitted[i] for i, _ in ms_placed]), np.concatenate([c for _, c in ms_placed]))
    if reference_pose is None or reference_pose[2] > 2.0:
      continue
    R_true, t = reference_pose[0], reference_pose[1]
    for face, ms in groups.items():
      if face in placement:
        continue
      for i, c in ms:
        errs = [
          (np.linalg.norm(cv2.projectPoints(obj[i] @ G.T, cv2.Rodrigues(R_true)[0], t, K, dist)[0].reshape(-1, 2) - c, axis=1).mean(), key(G))
          for G in CUBE_GROUP
        ]
        e, k = min(errs)
        if e < 3.0:
          single[face][k] += 1
  for face, counter in single.items():
    placement[face] = np.array(counter.most_common(1)[0][0], float).reshape(3, 3)

  report = {
    "reference": a.reference,
    "cube_edge_mm_anchor": a.cube_edge_mm,
    "fitted_marker_edge_mm": round(edge_m * 1000, 3),
    "fitted_marker_offset_mm": round(offset_m * 1000, 3),
    "print_scale_vs_pdf": [round(float(fit.x[0]), 4), round(float(fit.x[1]), 4)],
    "faces": {},
  }
  for face in cube["faces_config"]:
    if face not in placement:
      report["faces"][face] = "UNRESOLVED: capture this face together with a neighbouring face"
      continue
    M = placement[face]
    n_model = np.array(cube["face_axes"][face]["center"], float)
    physical = next(f for f, ax in cube["face_axes"].items() if np.allclose(ax["center"], M @ n_model))
    source = "pairwise" if face not in single else f"single-marker votes {dict(single[face])}"
    report["faces"][face] = {
      "sits_on_model_face": physical,
      "in_plane_rotation_deg": round(angle(M) if physical == face else float("nan"), 1),
      "M": np.rint(M).astype(int).tolist(),
      "evidence": source,
    }
  occupied = Counter(v["sits_on_model_face"] for v in report["faces"].values() if isinstance(v, dict))
  report["conflicts"] = [f for f, n in occupied.items() if n > 1]

  board = {"family": a.family, "faces": []}
  for f in build_board(cube_json, a.family, edge_m, offset_m, half)["faces"]:
    M = placement.get(face_of[f["id"]])
    if M is None:
      continue
    board["faces"].append(
      {
        "id": f["id"],
        "corners_object_m": (np.array(f["corners_object_m"]) @ M.T).tolist(),
        "normal_object": (M @ np.array(f["normal_object"])).tolist(),
        "center_object_m": (M @ np.array(f["center_object_m"])).tolist(),
      }
    )
  fixed = {f["id"]: np.array(f["corners_object_m"]) for f in board["faces"]}

  obj = {f["id"]: np.array(f["corners_object_m"]) for f in model["faces"]}
  # Validation: joint PnP over every visible face, model board vs measured board.
  rows = []
  for png, groups, _ in frames:
    ms = [(i, c) for g in groups.values() for i, c in g]
    if len({face_of[i] for i, _ in ms}) < 2:
      continue
    row = {"frame": f"{png.parent.name}/{png.name}"}
    for name, table in (("model", obj), ("measured", fixed)):
      use = [(i, c) for i, c in ms if i in table]
      s = solve(np.concatenate([table[i] for i, _ in use]), np.concatenate([c for _, c in use]))
      row[name + "_px"] = round(s[2], 2) if s else None
    rows.append(row)
  report["multi_face_validation"] = rows
  print(json.dumps(report, indent=2))
  if report["conflicts"]:
    raise SystemExit(f"Conflicting placements {report['conflicts']}; board not written")
  a.output.write_text(json.dumps(board, indent=2))
  print(f"Wrote {len(board['faces'])} markers to {a.output}")


if __name__ == "__main__":
  main()
