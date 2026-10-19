"""Overlay the hand model (measured joint angles) on the live wrist-camera image.

Read-only: subscribes to the wujihandros2 joint states and opens the RealSense; never
commands the hand. The hand model is posed with the measured joints and projected with
the release camera extrinsics (policy_config.json camera_to_tag / tag_in_palm) and the
live intrinsics. If the dots sit on the real fingers the extrinsics are right; a uniform
shift means the real camera mount differs from the trained one, which biases the cube
position the policy sees. The cross marks the sim reset cube centre.

--run DIR --times 4 50: offline, on a recorded run (wrist.mp4 + logged joint angles).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_handover import TAG_IN_PALM_POS, HandLink, camera_extrinsics  # noqa: E402

COLORS = {"palm": (160, 160, 160), "finger1": (0, 0, 255), "finger2": (0, 200, 0), "finger3": (255, 80, 0),
          "finger4": (0, 220, 255), "finger5": (255, 0, 255)}  # BGR


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--joint-state-topic", default="/right_hand/joint_states")
  p.add_argument("--output", type=Path, default=ROOT / "logs/camera_extrinsics_check")
  p.add_argument("--run", type=Path, help="Recorded run directory (offline mode)")
  p.add_argument("--times", type=float, nargs="+", default=[0.0], help="Seconds into the run (offline mode)")
  p.add_argument("--release-extrinsics", action="store_true", help="Check the release extrinsics, not the measured ones")
  a = p.parse_args()
  if a.run is not None:
    offline(a)
    return

  import pyrealsense2 as rs

  cfg = json.loads((ROOT / "release_models/policy_config.json").read_text())
  calib = json.loads(
    (ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json").read_text()
  )
  # palm -> optical camera: p_cam = Rcam^T (T^T (p_palm - P) - pcam)
  R_pc, t_pc = extrinsics(cfg, a.release_extrinsics)

  hand = HandLink("right", a.joint_state_topic, publish=False, driver_only=True)
  deadline = time.monotonic() + 5
  q = None
  while q is None and time.monotonic() < deadline:
    q, _ = hand.joints()
    time.sleep(0.05)
  if q is None:
    hand.close()
    raise SystemExit(f"No joint states on {a.joint_state_topic} (is the wujihand driver running?)")

  pipeline = rs.pipeline()
  rcfg = rs.config()
  rcfg.enable_device(calib["serial"])
  rcfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
  profile = pipeline.start(rcfg)
  try:
    for _ in range(30):
      pipeline.wait_for_frames(2000)
    rgb = np.asanyarray(pipeline.wait_for_frames(2000).get_color_frame().get_data()).copy()
    q, _ = hand.joints()
  finally:
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipeline.stop()
    hand.close()
  K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
  dist = np.array(intr.coeffs, dtype=np.float64)

  overlay = draw_overlay(cfg, q, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), K, dist, R_pc, t_pc)
  out = a.output / time.strftime("%Y%m%d_%H%M%S")
  out.mkdir(parents=True)
  cv2.imwrite(str(out / "raw.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
  cv2.imwrite(str(out / "overlay.png"), overlay)
  (out / "joints.json").write_text(json.dumps(dict(zip(cfg["joint_names"], np.round(q, 4).tolist())), indent=2))
  print(f"Saved {out / 'overlay.png'}")


def extrinsics(cfg, use_release=False):
  Rcam, pcam, _ = camera_extrinsics(cfg, use_release)
  T_tag = np.array(cfg["tag_in_palm_rotation"])
  return Rcam.T @ T_tag.T, -Rcam.T @ (T_tag.T @ TAG_IN_PALM_POS + pcam)


def offline(a):
  cfg = json.loads((ROOT / "release_models/policy_config.json").read_text())
  calib = json.loads(
    (ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json").read_text()
  )
  K = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1.0]])
  dist = np.array(calib["coeffs"])
  R_pc, t_pc = extrinsics(cfg, a.release_extrinsics)
  d = np.load(a.run / "trajectory.npz")
  t = d["t"] - d["t"][0]
  video = cv2.VideoCapture(str(a.run / "wrist.mp4"))
  out = a.output / a.run.name
  out.mkdir(parents=True, exist_ok=True)
  for sec in a.times:
    i = int(np.argmin(np.abs(t - sec)))
    video.set(cv2.CAP_PROP_POS_FRAMES, int(d["frame"][i]))
    ok, frame = video.read()
    if not ok:
      continue
    overlay = draw_overlay(cfg, d["q"][i], frame, K, dist, R_pc, t_pc)
    cv2.putText(overlay, f"t={t[i]:.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(out / f"overlay_{t[i]:06.1f}s.png"), overlay)
    print(f"Saved {out / f'overlay_{t[i]:06.1f}s.png'}")


def draw_overlay(cfg, q, overlay, K, dist, R_pc, t_pc):
  overlay = overlay.copy()
  m = mujoco.MjModel.from_xml_path(str(ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_mjlab.xml"))
  d = mujoco.MjData(m)
  for name, value in zip(cfg["joint_names"], q):
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = value
  mujoco.mj_forward(m, d)

  for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 1:
      continue
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    part = "palm" if "palm" in body else body.split("_")[1]
    mesh = m.geom_dataid[g]
    verts = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
    verts = verts[:: max(1, len(verts) // 400)]
    palm_pts = d.geom_xpos[g] + verts @ d.geom_xmat[g].reshape(3, 3).T
    cam_pts = palm_pts @ R_pc.T + t_pc
    cam_pts = cam_pts[cam_pts[:, 2] > 0.01]
    if len(cam_pts) == 0:
      continue
    px = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    for u, v in px:
      if 0 <= u < 640 and 0 <= v < 480:
        cv2.circle(overlay, (int(u), int(v)), 1, COLORS[part], -1)
  nominal_palm = np.array([cfg["cube_edge_m"] / 2 + 0.003, 0.009, 0.105])
  c = cv2.projectPoints((R_pc @ nominal_palm + t_pc)[None], np.zeros(3), np.zeros(3), K, dist)[0].reshape(2)
  cv2.drawMarker(overlay, (int(c[0]), int(c[1])), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
  cv2.putText(overlay, "model: thumb red, index green, middle blue, ring yellow, pinky magenta; + = sim cube centre",
              (6, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
  return overlay


if __name__ == "__main__":
  main()
