"""Capture wrist-camera frames and check printed cube stickers against the PnP board.

Read-only: opens the RealSense only, never talks to the hand. For each saved frame
it reports both marker families seen, the face/slot each ID should occupy, and the
per-marker reprojection error under the board pose (a mis-rotated sticker shows up
as a large error or is dropped by RANSAC). Annotated PNGs: green outline = marker
consistent with the board pose, red = inconsistent, yellow = not used; the dot marks
each marker's corner 0; axes X red / Y green / Z blue are the cube frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Sticker placement and print scale measured on the physical cube (calibrate_cube_board.py).
MEASURED_BOARD = ROOT / "hardware/palm_down_cube_45mm/board_measured_aruco4x4.json"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_handover import RobustBoardEstimator, build_board  # noqa: E402

DICTS = {"aruco4x4_50": cv2.aruco.DICT_4X4_50, "tag36h11": cv2.aruco.DICT_APRILTAG_36h11}


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--family", default="aruco4x4_50", choices=list(DICTS))
  p.add_argument("--marker-edge-mm", type=float, default=10.8333)
  p.add_argument("--board", type=Path, default=MEASURED_BOARD if MEASURED_BOARD.exists() else None,
                 help="Measured board from calibrate_cube_board.py (overrides --family/--marker-edge-mm)")
  p.add_argument("--nominal-board", dest="board", action="store_const", const=None,
                 help="Ignore the measured board and use cube_tags45.json as designed")
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--interval", type=float, default=1.0)
  p.add_argument("--output", type=Path, default=ROOT / "logs/cube_tag_check")
  a = p.parse_args()

  import pyrealsense2 as rs

  calib = json.loads(
    (ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json").read_text()
  )
  cube = json.loads((ROOT / "hardware/palm_down_cube_45mm/cube_tags45.json").read_text())
  offset = 100 if a.family == "tag36h11" else 0
  slots = {int(i) + offset: f"{face}-{s}" for face, layout in cube["faces_config"].items() for i, s in layout.items()}
  out = a.output / time.strftime("%Y%m%d_%H%M%S")
  out.mkdir(parents=True)
  board = (
    json.loads(a.board.read_text())
    if a.board
    else build_board(ROOT / "hardware/palm_down_cube_45mm/cube_tags45.json", a.family, a.marker_edge_mm / 1000)
  )
  (out / "board.json").write_text(json.dumps(board, indent=2))
  corners_obj = {f["id"]: np.array(f["corners_object_m"]) for f in board["faces"]}
  estimator = RobustBoardEstimator(out / "board.json")
  detectors = {}
  for name, d in DICTS.items():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detectors[name] = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(d), params)

  pipeline = rs.pipeline()
  cfg = rs.config()
  cfg.enable_device(calib["serial"])
  cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
  profile = pipeline.start(cfg)
  intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
  dist = np.array(intr.coeffs, dtype=np.float64)
  reports = []
  try:
    for _ in range(30):  # let auto-exposure settle
      pipeline.wait_for_frames(2000)
    print(f"Capturing {a.seconds:.0f} s into {out}", flush=True)
    end = time.monotonic() + a.seconds
    index = 0
    while time.monotonic() < end:
      rgb = np.asanyarray(pipeline.wait_for_frames(2000).get_color_frame().get_data()).copy()
      gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
      seen = {}
      for name, det in detectors.items():
        c, ids, _ = det.detectMarkers(gray)
        seen[name] = ([] if ids is None else ids.flatten().tolist(), c)
      ids, corners = seen[a.family]
      pose = estimator.estimate(rgb, K, dist)
      img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
      markers = []
      for tid, c in zip(ids, corners):
        c = c[0]
        err = None
        if pose is not None and tid in corners_obj:
          proj = cv2.projectPoints(corners_obj[tid], cv2.Rodrigues(pose.rotation)[0], pose.translation, K, dist)[0].reshape(-1, 2)
          err = float(np.linalg.norm(proj - c, axis=1).mean())
        used = pose is not None and tid in pose.ids
        color = (0, 255, 255) if not used else (0, 200, 0) if err is not None and err < 3 else (0, 0, 255)
        cv2.polylines(img, [c.astype(int)], True, color, 2)
        cv2.circle(img, tuple(c[0].astype(int)), 4, (255, 0, 255), -1)
        label = f"{tid} {slots.get(tid, '?')}" + (f" {err:.1f}px" if err is not None else "")
        cv2.putText(img, label, tuple((c.mean(0) + [-25, -12]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        markers.append({"id": tid, "slot": slots.get(tid), "used": used, "reproj_px": err})
      if pose is not None:
        cv2.drawFrameAxes(img, K, dist, cv2.Rodrigues(pose.rotation)[0], pose.translation, 0.035, 2)
      header = (
        f"{a.family}: {len(ids)} | other family: {len(seen['tag36h11' if a.family == 'aruco4x4_50' else 'aruco4x4_50'][0])}"
        + (f" | pose OK, {len(pose.ids)} used, {pose.reprojection_error_px:.2f}px, z={pose.translation[2] * 100:.1f}cm" if pose else " | no pose")
      )
      cv2.putText(img, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      cv2.imwrite(str(out / f"frame{index:02d}_raw.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
      cv2.imwrite(str(out / f"frame{index:02d}_annotated.png"), img)
      report = {
        "frame": index,
        "ids_by_family": {k: v[0] for k, v in seen.items()},
        "pose_found": pose is not None,
        "pose_ids": list(pose.ids) if pose else [],
        "pose_reproj_px": pose.reprojection_error_px if pose else None,
        "markers": markers,
      }
      reports.append(report)
      print(json.dumps({k: report[k] for k in ("frame", "ids_by_family", "pose_found", "pose_ids")}), flush=True)
      index += 1
      time.sleep(max(a.interval - 0.04, 0))
  finally:
    pipeline.stop()
    (out / "report.json").write_text(json.dumps(reports, indent=2))
    print(f"Saved {len(reports)} frames to {out}", flush=True)


if __name__ == "__main__":
  main()
