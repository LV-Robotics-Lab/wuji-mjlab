"""Render a demo video of a real handover run: synchronized sim mirror + wrist camera + state machine.

Input is a run directory written by teleop_handover.py (trajectory.npz, optional wrist.mp4,
events.json, args.json). Nothing here talks to hardware.

  left   sim mirror: the trained hand MJCF posed with the MEASURED joint angles, the cube at
         its VISUAL pose (faces coloured by cube axis: +X red, +Y green, +Z blue, opposite
         faces lighter), the goal orientation as a translucent cube above it
  right  real wrist camera with the detected cube axes (thick) and goal axes (thin); for runs
         without wrist.mp4, the sim rendered from the calibrated wrist camera instead
  top    phase machine TELEOP/HOLD/OPEN/CLOSE/POLICY, policy sub-state, goal error, goals

Rendering is offscreen through GLFW on an X display (EGL is unavailable on this machine),
so it uses DISPLAY=:1 (the workstation session) when none is set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASES = ["TELEOP", "HOLD", "OPEN", "CLOSE", "POLICY"]
SUBSTATES = ["APPROACH", "ON TARGET", "GOAL REACHED", "NO VISION"]
FACE_COLORS = {  # cube-frame face normal -> RGBA
  (1, 0, 0): (0.90, 0.20, 0.20), (-1, 0, 0): (1.00, 0.65, 0.65),
  (0, 1, 0): (0.20, 0.75, 0.25), (0, -1, 0): (0.65, 0.95, 0.65),
  (0, 0, 1): (0.20, 0.35, 0.95), (0, 0, -1): (0.65, 0.75, 1.00),
}


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("run_dir", type=Path)
  p.add_argument("--output", type=Path, help="Default: <run_dir>/demo.mp4")
  p.add_argument("--start", type=float, default=0.0, help="Seconds from the first logged tick")
  p.add_argument("--end", type=float, default=None)
  p.add_argument("--policy-only", action="store_true", help="Only the POLICY phase (+1 s around it)")
  p.add_argument("--gl", default=os.environ.get("MUJOCO_GL", "glfw"), choices=["glfw", "egl", "osmesa"])
  return p.parse_args()


a = parse_args()
os.environ["MUJOCO_GL"] = a.gl
if a.gl == "glfw" and not os.environ.get("DISPLAY"):
  os.environ["DISPLAY"] = ":1"

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_handover import TAG_IN_PALM_POS  # noqa: E402

PALM_WORLD_ROT = Rotation.from_euler("y", 90, degrees=True).as_matrix()  # palm-down, as in training
PALM_WORLD_POS = np.array([0.0, 0.0, 0.35])
GOAL_OFFSET = np.array([0.0, 0.0, 0.09])  # goal ghost above the cube, like the official demo


def run_extrinsics(cfg):
  """Extrinsics the run logged its cube poses with (older runs: the release ones)."""
  path = a.run_dir / "extrinsics.json"
  e = json.loads(path.read_text()) if path.exists() else cfg
  return np.array(e["camera_to_tag_rotation"]), np.array(e["camera_origin_tag_m"])


def derive_fields(d, cfg, args_json):
  """Phase / goal counters for logs written before teleop_handover logged them."""
  n = len(d["t"])
  if "phase" in d.files:
    return d["phase"], d["goal_index"], d["goals_reached"]
  state = d["state"]
  q_default = np.array(cfg["default_joint_pos"])
  phase = np.array([{0: 0, 1: 1, 3: 4}.get(int(s), 2) for s in state])
  i = 0
  while i < n:
    if state[i] == 2:
      j = i
      while j < n and state[j] == 2:
        j += 1
      if np.abs(d["target"][j - 1] - q_default).max() < 1e-3:
        phase[i:j] = 3
      i = j
    else:
      i += 1
  timeout = float(args_json.get("goal_timeout_s", 8.0))
  goal_index = np.zeros(n, int)
  reached = np.zeros(n, int)
  k = r = 0
  start_t = None
  for i in range(n):
    if phase[i] == 4:
      if start_t is None:
        k, start_t = k + 1, d["t"][i]
      elif np.abs(d["goal"][i] - d["goal"][i - 1]).max() > 1e-6:
        r += int(d["t"][i] - start_t < timeout - 0.1)
        k, start_t = k + 1, d["t"][i]
    else:
      start_t = None
    goal_index[i], reached[i] = k, r
  return phase, goal_index, reached


def build_model(cfg):
  spec = mujoco.MjSpec.from_file(str(ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_mjlab.xml"))
  spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
  spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 960
  palm = spec.body("right_palm_link")
  palm.pos = PALM_WORLD_POS
  palm.quat = Rotation.from_matrix(PALM_WORLD_ROT).as_quat()[[3, 0, 1, 2]]
  spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.42, 0.52, 0.66], rgb2=[0.06, 0.08, 0.12], width=512, height=3072)
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[0.16, 0.24, 0.34], rgb2=[0.11, 0.17, 0.25], width=512, height=512)
  floor_mat = spec.add_material(name="grid", texrepeat=[8, 8], reflectance=0.1)
  floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], material="grid", group=0)
  spec.worldbody.add_light(pos=[0.2, -0.3, 1.0], dir=[-0.2, 0.3, -1.0], diffuse=[0.6, 0.6, 0.6])
  half = cfg["cube_edge_m"] / 2
  for name, alpha in (("cube", 1.0), ("goal", 0.35)):
    body = spec.worldbody.add_body(name=name, mocap=True)
    body.add_geom(name=f"{name}_core", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half - 0.0006] * 3,
                  rgba=[0.92, 0.92, 0.92, alpha * 0.6], contype=0, conaffinity=0, group=0)
    for normal, rgb in FACE_COLORS.items():
      n = np.array(normal, float)
      z = n
      x = np.array([0, 0, 1.0]) if abs(n[2]) < 0.5 else np.array([1.0, 0, 0])
      x = x - z * (x @ z)
      x /= np.linalg.norm(x)
      R = np.column_stack([x, np.cross(z, x), z])
      body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half * 0.82, half * 0.82, 0.0006], pos=n * half,
                    quat=Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]], rgba=[*rgb, alpha],
                    contype=0, conaffinity=0, group=0)
  # Calibrated wrist camera (used when the run has no wrist recording).
  Rcam, pcam = run_extrinsics(cfg)
  T_tag = np.array(cfg["tag_in_palm_rotation"])
  R_palm_optical = T_tag @ Rcam
  cam_pos = T_tag @ pcam + TAG_IN_PALM_POS
  R_mj = R_palm_optical @ np.diag([1, -1, -1])  # MuJoCo cameras look along -z with +y up
  fy = np.array(cfg["camera_K"])[1, 1]
  palm.add_camera(name="wrist", pos=cam_pos, quat=Rotation.from_matrix(R_mj).as_quat()[[3, 0, 1, 2]],
                  fovy=float(np.degrees(2 * np.arctan(240 / fy))))
  m = spec.compile()
  return m


def put(img, text, xy, scale=0.6, color=(255, 255, 255), thick=1):
  cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
  run = a.run_dir
  d = np.load(run / "trajectory.npz")
  cfg = json.loads((ROOT / "release_models/policy_config.json").read_text())
  args_json = json.loads((run / "args.json").read_text()) if (run / "args.json").exists() else {}
  events = json.loads((run / "events.json").read_text()) if (run / "events.json").exists() else []
  phase, goal_index, reached = derive_fields(d, cfg, args_json)
  t = d["t"] - d["t"][0]
  success_rad = float(args_json.get("success_rad", 0.2))
  success_ticks = int(args_json.get("success_ticks", 5))
  stop_after = int(args_json.get("stop_after_goals", 0) or 0)
  calib = json.loads(
    (ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json").read_text()
  )
  K = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1.0]])
  dist = np.array(calib["coeffs"])
  Rcam, pcam = run_extrinsics(cfg)
  T_tag = np.array(cfg["tag_in_palm_rotation"])

  idx = np.arange(len(t))
  sel = t >= a.start
  if a.end is not None:
    sel &= t <= a.end
  if a.policy_only and (phase == 4).any():
    pol = idx[phase == 4]
    sel &= (t >= t[pol[0]] - 1.0) & (t <= t[pol[-1]] + 1.0)
  ticks = idx[sel]
  if len(ticks) == 0:
    raise SystemExit("No ticks in the selected range")

  m = build_model(cfg)
  dsim = mujoco.MjData(m)
  qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in cfg["joint_names"]]
  cube_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")]
  goal_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "goal")]
  cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
  cube_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == cube_body]
  cube_alpha = m.geom_rgba[cube_geoms, 3].copy()
  W, H = 960, 720
  ext = mujoco.Renderer(m, H, W)
  wrist_sim = mujoco.Renderer(m, 480, 640)
  opt = mujoco.MjvOption()
  opt.geomgroup[:] = 0
  opt.geomgroup[0] = opt.geomgroup[1] = 1
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  nominal_palm = np.array([cfg["cube_edge_m"] / 2 + 0.003, 0.009, 0.105])
  cam.lookat[:] = PALM_WORLD_POS + PALM_WORLD_ROT @ nominal_palm + np.array([0, 0, 0.03])
  cam.distance, cam.azimuth, cam.elevation = 0.34, 120.0, -15.0

  video = cv2.VideoCapture(str(run / "wrist.mp4")) if (run / "wrist.mp4").exists() else None
  has_frames = "frame" in d.files and video is not None and video.isOpened()
  fps = round(1 / float(np.median(np.diff(d["t"])))) if len(t) > 1 else 20
  out_path = a.output or run / "demo.mp4"
  header_h = 150
  size = (2 * W, H + header_h)
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg:
    enc = subprocess.Popen(
      [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
       "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_path)],
      stdin=subprocess.PIPE,
    )
    write = enc.stdin.write
  else:
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    write = lambda b: writer.write(np.frombuffer(b, np.uint8).reshape(size[1], size[0], 3))

  hold = 0
  reached_at = -np.inf
  try:
    for n, i in enumerate(ticks):
      # ---- state ----
      usable = bool(d["usable"][i])
      in_policy = phase[i] == 4
      err = float(Rotation.from_matrix(d["rot"][i] @ d["goal"][i].T).magnitude()) if in_policy and usable else None
      hold = hold + 1 if err is not None and err < success_rad else 0
      if i > 0 and reached[i] > reached[i - 1]:
        reached_at = t[i]
      if not in_policy:
        sub = None
      elif t[i] - reached_at < 1.0:
        sub = "GOAL REACHED"
      elif not usable:
        sub = "NO VISION"
      elif hold > 0:
        sub = "ON TARGET"
      else:
        sub = "APPROACH"

      # ---- sim mirror ----
      dsim.qpos[qadr] = d["q"][i]
      R_cube_palm = T_tag @ d["rot"][i]
      p_cube_palm = T_tag @ d["pos"][i] + TAG_IN_PALM_POS
      p_world = PALM_WORLD_POS + PALM_WORLD_ROT @ p_cube_palm
      seen = (d["usable"][: i + 1]).any()
      dsim.mocap_pos[cube_mocap] = p_world if seen else [0, 0, -1]
      dsim.mocap_quat[cube_mocap] = Rotation.from_matrix(PALM_WORLD_ROT @ R_cube_palm).as_quat()[[3, 0, 1, 2]]
      m.geom_rgba[cube_geoms, 3] = cube_alpha * (1.0 if usable else 0.35)
      if in_policy:
        dsim.mocap_pos[goal_mocap] = p_world + GOAL_OFFSET
        goal_world = PALM_WORLD_ROT @ T_tag @ d["goal"][i]
        dsim.mocap_quat[goal_mocap] = Rotation.from_matrix(goal_world).as_quat()[[3, 0, 1, 2]]
      else:
        dsim.mocap_pos[goal_mocap] = [0, 0, -1]
      mujoco.mj_forward(m, dsim)
      ext.update_scene(dsim, camera=cam, scene_option=opt)
      left = cv2.cvtColor(ext.render(), cv2.COLOR_RGB2BGR)
      put(left, "SIM MIRROR: measured joints + camera cube pose", (14, 30), 0.65)
      put(left, "lower cube = observed   upper translucent cube = goal", (14, 58), 0.55, (210, 210, 210))

      # ---- wrist view ----
      if has_frames and d["frame"][i] >= 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(d["frame"][i]))
        ok, frame = video.read()
        wrist = frame if ok else np.zeros((480, 640, 3), np.uint8)
        label = "WRIST CAMERA (real)"
      else:
        wrist_sim.update_scene(dsim, camera="wrist", scene_option=opt)
        wrist = cv2.cvtColor(wrist_sim.render(), cv2.COLOR_RGB2BGR)
        label = "WRIST CAMERA VIEW (sim; run has no recording)"
      if usable:
        R_c = Rcam.T @ d["rot"][i]
        t_c = Rcam.T @ (d["pos"][i] - pcam)
        cv2.drawFrameAxes(wrist, K, dist, cv2.Rodrigues(R_c)[0], t_c, 0.035, 3)
        if in_policy:
          cv2.drawFrameAxes(wrist, K, dist, cv2.Rodrigues(Rcam.T @ d["goal"][i])[0], t_c, 0.022, 1)
      right = cv2.resize(wrist, (W, H))
      put(right, label, (14, 30), 0.65)
      put(right, "axes: X red  Y green  Z blue   thick = observed  thin = goal", (14, 58), 0.55, (220, 220, 220))

      # ---- header ----
      header = np.full((header_h, 2 * W, 3), (32, 20, 12), np.uint8)
      put(header, "Wuji hand | palm-down 45 mm cube reorientation | wrist-camera RL student on real hardware",
          (20, 30), 0.7)
      for k, name in enumerate(PHASES):
        x0 = 20 + k * 190
        active = PHASES[phase[i]] == name
        cv2.rectangle(header, (x0, 45), (x0 + 175, 80), (170, 110, 35) if active else (58, 45, 35), -1)
        put(header, name, (x0 + 14, 70), 0.6, (255, 255, 255) if active else (150, 150, 150))
      for k, name in enumerate(SUBSTATES):
        x0 = 1000 + k * 225
        active = sub == name
        color = (80, 145, 30) if name == "GOAL REACHED" else (35, 90, 200) if name == "NO VISION" else (170, 110, 35)
        cv2.rectangle(header, (x0, 45), (x0 + 210, 80), color if active else (58, 45, 35), -1)
        text = f"{name} {min(hold, success_ticks)}/{success_ticks}" if name == "ON TARGET" else name
        put(header, text, (x0 + 12, 70), 0.55, (255, 255, 255) if active else (150, 150, 150))
      goals = f"{reached[i]}/{stop_after}" if stop_after else f"{reached[i]}"
      line = f"t = {t[i]:6.1f} s   |   goals reached {goals}"
      if in_policy:
        line += f"   |   goal {goal_index[i]}   |   " + (
          f"visual error {np.degrees(err):5.1f} deg (success < {np.degrees(success_rad):.1f} deg)"
          if err is not None else "pose unavailable: holding last action")
      put(header, line, (20, 118), 0.65)
      recent = [e for et, e in events if 0 <= t[i] - et < 2.5]
      if recent:
        width = cv2.getTextSize(recent[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        put(header, recent[-1], (2 * W - 20 - width, 118), 0.6, (120, 220, 255))
      canvas = np.vstack([header, np.hstack([left, right])])
      write(canvas.tobytes())
      if n % 200 == 0:
        print(f"{n}/{len(ticks)} frames", flush=True)
  finally:
    if ffmpeg:
      enc.stdin.close()
      enc.wait()
    else:
      writer.release()
    ext.close()
    wrist_sim.close()
    m.geom_rgba[cube_geoms, 3] = cube_alpha
  print(f"Wrote {out_path} ({len(ticks)} frames at {fps} fps)")


if __name__ == "__main__":
  main()
