"""Manus pre-grasp -> hold -> vision gate -> wrist-camera student reorientation.

--driver-only: no glove/controller; run only the wujihandros2 driver (it holds the hand's
startup pose) and this script publishes targets straight to /right_hand/joint_commands:
  ros2 launch wujihand_bringup wujihand.launch.py hand_name:=right_hand serial_number:=387B366A3533 \
    publish_rate:=500.0 filter_cutoff_freq:=3.0 diagnostics_rate:=2.0
Otherwise it runs beside the existing hand teleop chain, which owns the hand:
  bash PrometheusV4/scripts/lib/manus_hand_chain.sh --hand-side right --filter-cutoff-hz 3.0
Takeover uses the controller's INFERENCE mode (/wuji_hand/<side>/switch_mode) and
publishes joint targets on /wuji_hand/<side>/joint_command. The controller applies
no smoothing or rate limit in INFERENCE mode, so all limits live here.

Modes:
  --vision-only  camera + cube PnP only, no ROS, no motor commands
  --shadow       full state machine on live data, never switches mode or publishes
  (default)      live handover

Scripted pre-grasp (matches the sim reset: hand at default_joint_pos, cube at the
nominal position): glove opens the hand -> h -> o -> push the cube up into the hand
following the "move" hint -> g -> release the cube -> p.

Keys (terminal, or the preview window with --gui):
  h      TELEOP -> HOLD: freeze the last teleop command and take over the hand
  o      HOLD: blend to the open pre-grasp pose (pregrasp_open_pose.json)
  g      HOLD: blend to the training grasp (default_joint_pos), closing around the cube
  p      HOLD -> POLICY (requires the vision gate unless --force-start)
  space  POLICY -> HOLD at the current target
  n      next goal
  t t    HOLD -> TELEOP (press twice within 2 s)
  q      hold and quit (controller stays in INFERENCE holding the last target)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
# Sticker placement and print scale measured on the physical cube (calibrate_cube_board.py).
MEASURED_BOARD = ROOT / "hardware/palm_down_cube_45mm/board_measured_aruco4x4.json"
# Training grasp widened for pushing the cube in from below (compute_open_pose.py).
OPEN_POSE = Path(__file__).resolve().parent / "pregrasp_open_pose.json"
# Real wrist-camera pose measured from glove silhouettes (calibrate_camera_extrinsics.py).
MEASURED_EXTRINSICS = Path(__file__).resolve().parent / "camera_extrinsics_measured.json"


def camera_extrinsics(cfg: dict, use_release: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
  """camera->tag rotation and camera origin in the tag frame (policy_config.json convention)."""
  if MEASURED_EXTRINSICS.exists() and not use_release:
    e = json.loads(MEASURED_EXTRINSICS.read_text())
    return np.array(e["camera_to_tag_rotation"]), np.array(e["camera_origin_tag_m"]), MEASURED_EXTRINSICS.name
  return np.array(cfg["camera_to_tag_rotation"]), np.array(cfg["camera_origin_tag_m"]), "policy_config.json (release)"
sys.path.insert(0, str(ROOT / "src"))
from wuji_mjlab.perception.apriltag import AprilTagBoardEstimator, BoardPose, PoseCache  # noqa: E402

JOINT_NAMES = [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
# src/wuji_mjlab/tasks/reorient/reorient_constants.py TAG_IN_PALM_POS, and the sim
# reset cube position object_palm_pos=(edge/2 + 0.003, 0.009, 0.105) from
# scripts/train/train_cube_downward.py:make_cfg.
TAG_IN_PALM_POS = np.array([0.0262, 0.0, -0.0563])
TELEOP, HOLD, BLEND, POLICY = "TELEOP", "HOLD", "BLEND", "POLICY"
STATE_CODES = {TELEOP: 0, HOLD: 1, BLEND: 2, POLICY: 3}
# Finer phase for logs/videos: the blend is split into its open and close directions.
PHASE_CODES = {"TELEOP": 0, "HOLD": 1, "OPEN": 2, "CLOSE": 3, "POLICY": 4}


class SensorHistory:
  """Verbatim copy of src/wuji_mjlab/perception/cube_wrist.py:SensorHistory.

  That module imports MuJoCo and the sim asset tree at import time.
  """

  def __init__(self):
    self.history = None

  def update(self, q, limits, target, last_action, position, rotation, goal):
    center = limits.mean(axis=1)
    half = (limits[:, 1] - limits[:, 0]) / 2
    angles = np.clip((q - center) / (half + 1e-6), -1, 1)
    error = angles - np.clip((target - center) / (half + 1e-6), -1, 1)
    ori = (rotation @ goal.T).reshape(9)[3:]
    terms = [angles, error, position, ori, last_action]
    if self.history is None:
      self.history = [np.tile(t, (3, 1)) for t in terms]
    else:
      for h, t in zip(self.history, terms):
        h[:-1] = h[1:]
        h[-1] = t
    return np.concatenate([h.ravel() for h in self.history]).astype(np.float32)


def build_board(
  cube_json: Path, family: str, marker_edge_m: float, offset_m: float | None = None, half_m: float = 0.02251
) -> dict:
  """Same marker geometry as scripts/check_cube_wrist_tags.py:build_cube."""
  cfg = json.loads(cube_json.read_text())
  offset = cfg["tag_center_offset"] if offset_m is None else offset_m
  half = half_m
  h = marker_edge_m / 2
  board = {"family": family, "faces": []}
  for face, layout in cfg["faces_config"].items():
    axes = cfg["face_axes"][face]
    n, u, v = (np.array(axes[k], float) for k in ("center", "u", "v"))
    angle = cfg["face_rotations"][face] // 90
    for tid, slot in layout.items():
      x, y = {"T": (0, offset), "R": (offset, 0), "B": (0, -offset), "L": (-offset, 0)}[
        slot
      ]
      center = n * half + u * x + v * y
      corners = np.array(
        [center - h * u + h * v, center + h * u + h * v, center + h * u - h * v, center - h * u - h * v]
      )
      board["faces"].append(
        {
          "id": int(tid) + (100 if family == "tag36h11" else 0),
          "corners_object_m": np.roll(corners, angle, axis=0).tolist(),
          "normal_object": n.tolist(),
          "center_object_m": center.tolist(),
        }
      )
  return board


class RobustBoardEstimator(AprilTagBoardEstimator):
  """Fit all known markers first; use the release RANSAC path only when that fails.

  On real frames cv2.solvePnPRansac(SQPNP) often returns a poor minimal model (e.g.
  5/16 inliers) although a single SQPNP fit over all markers is within 1 px, so whole
  frames were dropped. Acceptance checks are the parent's: positive depth, every
  face pointing at the camera, max reprojection error <= max_error_px.
  """

  def estimate(self, rgb, K, distortion=None):
    cv2 = self.cv2
    corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
      return None
    known = [(int(i), c[0]) for i, c in zip(ids.flatten(), corners) if int(i) in self.faces]
    if len(known) >= 2:
      object_points = np.concatenate([self.faces[i]["corners_object_m"] for i, _ in known]).astype(np.float64)
      image_points = np.concatenate([c for _, c in known]).astype(np.float64)
      K = np.asarray(K, dtype=np.float64)
      dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
      try:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_SQPNP)
      except cv2.error:
        ok = False
      candidates = []
      for rv, tv in zip(rvecs, tvecs) if ok else []:
        rot, translation = cv2.Rodrigues(rv)[0], tv.flatten()
        if ((rot @ object_points.T).T + translation)[:, 2].min() <= 0:
          continue
        if any(
          np.dot(rot @ np.array(self.faces[i]["normal_object"]), rot @ np.array(self.faces[i]["center_object_m"]) + translation) >= 0
          for i, _ in known
        ):
          continue
        errors = np.linalg.norm(cv2.projectPoints(object_points, rv, tv, K, dist)[0].reshape(-1, 2) - image_points, axis=1)
        if errors.max() <= self.max_error_px:
          candidates.append(BoardPose(rot, translation, tuple(i for i, _ in known), float(errors.mean())))
      if candidates:
        return min(candidates, key=lambda p: p.reprojection_error_px)
    return super().estimate(rgb, K, distortion)


class Camera:
  """RealSense color stream + cube PnP in a background thread (latest result only)."""

  def __init__(self, serial: str, calib: dict, estimator: AprilTagBoardEstimator):
    import pyrealsense2 as rs

    self.estimator = estimator
    self.pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = self.pipeline.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    live = np.array([intr.fx, intr.fy, intr.ppx, intr.ppy])
    expected = np.array([calib[k] for k in ("fx", "fy", "cx", "cy")])
    if not np.allclose(live, expected, atol=1e-3):
      self.pipeline.stop()
      raise RuntimeError(f"Live intrinsics {live} differ from release calibration {expected}")
    self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
    # Factory model is (inverse) Brown-Conrady k1 k2 p1 p2 k3: forward projection
    # matches OpenCV's 5-coefficient model, so PnP uses the raw image directly.
    self.dist = np.array(intr.coeffs, dtype=np.float64)
    self.lock = threading.Lock()
    self.latest = (0, -np.inf, None, None)  # seq, monotonic time, pose, rgb
    self.detect_ms = 0.0
    self.stop_event = threading.Event()
    self.thread = threading.Thread(target=self._run, name="camera", daemon=True)
    self.thread.start()

  def _run(self):
    seq = 0
    while not self.stop_event.is_set():
      try:
        frames = self.pipeline.wait_for_frames(1000)
      except RuntimeError:
        continue
      color = frames.get_color_frame()
      if not color:
        continue
      stamp = time.monotonic()
      rgb = np.asanyarray(color.get_data()).copy()
      started = time.perf_counter()
      pose = self.estimator.estimate(rgb, self.K, self.dist)
      seq += 1
      with self.lock:
        self.detect_ms = (time.perf_counter() - started) * 1000
        self.latest = (seq, stamp, pose, rgb)

  def get(self):
    with self.lock:
      return self.latest

  def close(self):
    self.stop_event.set()
    self.thread.join(timeout=2)
    self.pipeline.stop()


class HandLink:
  """ROS side: measured joints, teleop commands, mode switch and target publishing."""

  def __init__(self, side: str, joint_state_topic: str, publish: bool, driver_only: bool = False):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_srvs.srv import SetBool

    self.rclpy = rclpy
    self.JointState = JointState
    self.SetBool = SetBool
    from rclpy.signals import SignalHandlerOptions

    # Keep Ctrl-C as a Python KeyboardInterrupt in the main thread; rclpy's own handler
    # would shut the context first and the final hold target could not be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    self.node = rclpy.create_node("wrist_student_handover")
    self.lock = threading.Lock()
    self.q = None
    self.q_time = -np.inf
    self.teleop_cmd = None
    self.teleop_cmd_time = -np.inf
    self.inference = False
    self._index_cache: dict[tuple, np.ndarray | None] = {}
    self.error = ""
    self.driver_only = driver_only
    # driver-only: the wujihandros2 driver's own command topic, next to its joint_states.
    cmd_topic = joint_state_topic.rsplit("/", 1)[0] + "/joint_commands" if driver_only else f"/wuji_hand/{side}/joint_command"
    self.node.create_subscription(
      JointState, joint_state_topic, self._on_joint_state, qos_profile_sensor_data
    )
    best_effort = QoSProfile(
      reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
    )
    if not driver_only:
      self.node.create_subscription(JointState, cmd_topic, self._on_command, best_effort)
    self.pub = None
    if publish:
      reliable = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1
      )
      self.pub = self.node.create_publisher(JointState, cmd_topic, reliable)
    self.switch_client = self.node.create_client(SetBool, f"/wuji_hand/{side}/switch_mode")
    self.executor = SingleThreadedExecutor()
    self.executor.add_node(self.node)
    self.thread = threading.Thread(target=self._spin, name="ros", daemon=True)
    self.thread.start()

  def _spin(self):
    from rclpy.executors import ExternalShutdownException

    try:
      self.executor.spin()
    except ExternalShutdownException:
      pass

  def _ordered(self, msg) -> np.ndarray | None:
    if len(msg.position) != 20:
      return None
    names = tuple(msg.name)
    if not names:
      return np.asarray(msg.position, dtype=np.float64)
    if names not in self._index_cache:
      # Driver: right_finger1_joint1..right_finger5_joint4 (1-based).
      # Teleop controller: thumb_joint_0..pinky_joint_3 (0-based), thumb = finger1.
      fingers = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5, "little": 5}
      index = np.full(20, -1)
      for i, name in enumerate(names):
        if m := re.search(r"finger(\d)_joint(\d)$", name):
          index[(int(m.group(1)) - 1) * 4 + int(m.group(2)) - 1] = i
        elif (m := re.search(r"(thumb|index|middle|ring|pinky|little)_joint_(\d)$", name)):
          index[(fingers[m.group(1)] - 1) * 4 + int(m.group(2))] = i
      self._index_cache[names] = None if (index < 0).any() else index
    if self._index_cache[names] is None:
      # Never raise in a ROS callback: it kills the executor thread.
      self.error = f"unrecognised joint names {names[:2]}..."
      return None
    return np.asarray(msg.position, dtype=np.float64)[self._index_cache[names]]

  def _on_joint_state(self, msg):
    q = self._ordered(msg)
    if q is not None:
      with self.lock:
        self.q, self.q_time = q, time.monotonic()

  def _on_command(self, msg):
    # In INFERENCE mode this topic carries our own targets; only teleop output is kept.
    if self.inference:
      return
    q = self._ordered(msg)
    if q is not None:
      with self.lock:
        self.teleop_cmd, self.teleop_cmd_time = q, time.monotonic()

  def joints(self):
    with self.lock:
      return (None if self.q is None else self.q.copy()), self.q_time

  def teleop_command(self):
    with self.lock:
      return (None if self.teleop_cmd is None else self.teleop_cmd.copy()), self.teleop_cmd_time

  def switch(self, inference: bool, timeout_s: float = 2.0):
    if self.driver_only:
      self.inference = inference
      return
    if not self.switch_client.wait_for_service(timeout_sec=timeout_s):
      raise RuntimeError("switch_mode service unavailable (is the teleop chain running?)")
    if inference:
      self.inference = True
    future = self.switch_client.call_async(self.SetBool.Request(data=inference))
    deadline = time.monotonic() + timeout_s
    while not future.done():
      if time.monotonic() > deadline:
        raise RuntimeError("switch_mode call timed out")
      time.sleep(0.002)
    result = future.result()
    if not result.success:
      raise RuntimeError(f"switch_mode failed: {result.message}")
    if not inference:
      self.inference = False

  def send(self, target: np.ndarray):
    if self.pub is None:
      return
    msg = self.JointState()
    msg.header.stamp = self.node.get_clock().now().to_msg()
    # Position-only, finger-major order (the order both the controller and the driver use).
    # Named commands are risky: the driver commands any joint whose name it cannot match to 0.
    msg.position = target.tolist()
    self.pub.publish(msg)

  def close(self):
    self.executor.shutdown()
    self.node.destroy_node()
    self.rclpy.try_shutdown()


class Keys:
  """Non-blocking single-key reader on the controlling terminal."""

  def __init__(self):
    self.enabled = sys.stdin.isatty()
    if self.enabled:
      self.saved = termios.tcgetattr(sys.stdin)
      tty.setcbreak(sys.stdin.fileno())

  def read(self) -> str:
    if not self.enabled:
      return ""
    out = b""
    while select.select([sys.stdin], [], [], 0)[0]:
      out += os.read(sys.stdin.fileno(), 64)
    return out.decode(errors="ignore")

  def close(self):
    if self.enabled:
      termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.saved)


def sample_goal(rng, current: np.ndarray, max_deg: float, min_deg: float) -> np.ndarray:
  if max_deg >= 180:
    return Rotation.random(random_state=rng).as_matrix()
  axis = rng.normal(size=3)
  axis /= np.linalg.norm(axis)
  angle = np.radians(rng.uniform(min_deg, max_deg))
  return Rotation.from_rotvec(axis * angle).as_matrix() @ current


def rotation_angle(R: np.ndarray) -> float:
  return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  mode = p.add_mutually_exclusive_group()
  mode.add_argument("--vision-only", action="store_true")
  mode.add_argument("--shadow", action="store_true")
  p.add_argument("--side", default="right", choices=["right"])
  p.add_argument("--driver-only", action="store_true",
                 help="No glove/controller: command the wujihandros2 driver directly ('h' holds the measured pose)")
  p.add_argument("--joint-state-topic", default="/right_hand/joint_states")
  p.add_argument("--family", default="aruco4x4_50", choices=["aruco4x4_50", "tag36h11"],
                 help="Must match the printed stickers (cube45_face_stickers_A4.pdf is DICT_4X4_50)")
  p.add_argument("--marker-edge-mm", type=float, default=10.8333,
                 help="Printed black marker edge (PDF states 10.8333 mm)")
  p.add_argument("--board", type=Path, default=MEASURED_BOARD if MEASURED_BOARD.exists() else None,
                 help="Measured board from calibrate_cube_board.py (overrides --family/--marker-edge-mm)")
  p.add_argument("--nominal-board", dest="board", action="store_const", const=None,
                 help="Ignore the measured board and use cube_tags45.json as designed")
  p.add_argument("--gui", action="store_true", help="OpenCV preview window")
  p.add_argument("--gate-ticks", type=int, default=10, help="Consecutive usable ticks before READY")
  p.add_argument("--nominal-tol-mm", type=float, default=25.0,
                 help="READY requires the cube within this distance of the sim reset position")
  p.add_argument("--drop-tol-mm", type=float, default=45.0,
                 help="POLICY -> HOLD when the cube is this far from the sim reset position")
  p.add_argument("--max-blind-s", type=float, default=1.0, help="POLICY -> HOLD after this long without usable vision")
  p.add_argument("--max-default-dev-rad", type=float, default=0.6,
                 help="Refuse POLICY when any held joint is farther than this from default_joint_pos")
  p.add_argument("--open-s", type=float, default=1.5, help="Duration of the 'o' blend to the open pre-grasp pose")
  p.add_argument("--close-s", type=float, default=2.0, help="Duration of the 'g' blend to default_joint_pos")
  p.add_argument("--max-joint-speed", type=float, default=4.0, help="rad/s cap on published targets in POLICY")
  p.add_argument("--startup-joint-speed", type=float, default=1.0, help="rad/s cap during the first --startup-s of POLICY")
  p.add_argument("--startup-s", type=float, default=1.0)
  p.add_argument("--max-goal-deg", type=float, default=90.0,
                 help="Goal = random rotation of at most this angle from the current cube pose; >=180 gives uniform SO(3)")
  p.add_argument("--min-goal-deg", type=float, default=30.0)
  p.add_argument("--goal-timeout-s", type=float, default=8.0)
  p.add_argument("--success-rad", type=float, default=0.2)
  p.add_argument("--success-ticks", type=int, default=5)
  p.add_argument("--force-start", action="store_true", help="Allow 'p' without the vision gate")
  p.add_argument("--stop-after-goals", type=int, default=0,
                 help="POLICY -> HOLD after this many goals are reached (0 = keep going)")
  p.add_argument("--no-record", action="store_true", help="Do not save the wrist camera video (wrist.mp4)")
  p.add_argument("--release-extrinsics", action="store_true",
                 help=f"Ignore {MEASURED_EXTRINSICS.name} and use the release camera extrinsics")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--log-dir", type=Path, default=ROOT / "logs/real_handover")
  a = p.parse_args()

  cfg = json.loads((ROOT / "release_models/policy_config.json").read_text())
  calib = json.loads(
    (ROOT / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json").read_text()
  )
  q_default = np.array(cfg["default_joint_pos"])
  limits = np.array(cfg["soft_joint_pos_limits"])
  Rcam, pcam, extrinsics_source = camera_extrinsics(cfg, a.release_extrinsics)
  T_tag = np.array(cfg["tag_in_palm_rotation"])
  ctrl_dt, scale, alpha = cfg["ctrl_dt"], cfg["action_scale"], cfg["ema_alpha"]
  edge = cfg["cube_edge_m"]
  nominal_palm = np.array([edge / 2 + 0.003, 0.009, 0.105])
  nominal_pos = T_tag.T @ (nominal_palm - TAG_IN_PALM_POS)
  q_open = None
  if OPEN_POSE.exists():
    open_cfg = json.loads(OPEN_POSE.read_text())
    if open_cfg["joint_names"] != cfg["joint_names"]:
      raise RuntimeError(f"{OPEN_POSE} joint order differs from policy_config.json")
    q_open = np.array(open_cfg["q_open"])

  run_dir = a.log_dir / time.strftime("%Y%m%d_%H%M%S")
  run_dir.mkdir(parents=True, exist_ok=False)
  board_path = run_dir / "board.json"
  if a.board is not None:
    board_path.write_text(a.board.read_text())
  else:
    board_path.write_text(
      json.dumps(build_board(ROOT / "hardware/palm_down_cube_45mm/cube_tags45.json", a.family, a.marker_edge_mm / 1000), indent=2)
    )
  (run_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))
  (run_dir / "extrinsics.json").write_text(json.dumps(
    {"source": extrinsics_source, "camera_to_tag_rotation": Rcam.tolist(), "camera_origin_tag_m": pcam.tolist()}, indent=2))

  torch.set_num_threads(1)
  policy = torch.jit.load(str(ROOT / "release_models/policy.jit"), map_location="cpu").eval()
  camera = Camera(calib["serial"], calib, RobustBoardEstimator(board_path))
  live = not (a.vision_only or a.shadow)
  hand = None if a.vision_only else HandLink(a.side, a.joint_state_topic, publish=live, driver_only=a.driver_only)
  if a.driver_only:
    print("driver-only: the driver holds the hand's startup pose; press h to take over", flush=True)
  keys = Keys()
  if not keys.enabled:
    print("stdin is not a TTY: keyboard control disabled", flush=True)
  if a.gui and not os.environ.get("DISPLAY"):
    # OpenCV's Qt backend aborts the process (uncatchable) without a display, e.g. over SSH.
    print("No DISPLAY (SSH session?): --gui disabled; the status line has the same information.\n"
          "To show the window on the workstation monitor instead, run with DISPLAY=:1", flush=True)
    a.gui = False
  if a.gui:
    import cv2

  rng = np.random.default_rng(a.seed)
  state = TELEOP
  cache = PoseCache(1, cfg["visual_timeout_s"], max_position_speed_m_s=0.5, max_rotation_speed_rad_s=12)
  last_seq = 0
  ready_ticks = 0
  target = None  # last target we published (or would publish in shadow mode)
  blend_from = blend_start = blend_to = None
  blend_s = 1.0
  blend_done = ""
  history = last = previous = goal = None
  policy_start = goal_start = 0.0
  goal_hold = goals_reached = goals_total = 0
  last_usable_time = -np.inf
  t_press = -np.inf
  log = {
    k: []
    for k in ("t", "state", "phase", "q", "target", "raw", "pos", "rot", "fresh", "usable", "goal", "obs",
              "goal_index", "goals_reached", "frame")
  }
  events = []
  blend_kind = "OPEN"
  recorder = None
  if not a.no_record:
    import cv2 as cv2_rec

    recorder = cv2_rec.VideoWriter(
      str(run_dir / "wrist.mp4"), cv2_rec.VideoWriter_fourcc(*"mp4v"), round(1 / ctrl_dt), (640, 480)
    )
  next_tick = time.monotonic()
  last_print = 0.0
  message = ""
  gui_keys = ""

  def enter_hold(reason: str):
    nonlocal state, message
    state = HOLD
    message = f"HOLD ({reason})"
    events.append((time.monotonic(), message))
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)

  print(f"Camera extrinsics: {extrinsics_source}", flush=True)
  print(f"Logging to {run_dir}\nBoard: {a.board or 'nominal cube_tags45.json (' + a.family + ')'}\n"
        f"Nominal cube position in tag frame: {np.round(nominal_pos * 1000, 1)} mm", flush=True)
  try:
    while True:
      now = time.monotonic()
      if now < next_tick:
        time.sleep(next_tick - now)
        now = time.monotonic()
      next_tick = max(next_tick + ctrl_dt, now)

      # Vision: feed each detection to the cache exactly once (sim: one render per control step).
      seq, stamp, pose, rgb = camera.get()
      new_pose = pose if seq != last_seq else None
      last_seq = seq
      fresh, usable, _ = cache.update([new_pose], now, Rcam, pcam)
      fresh, usable = bool(fresh[0]), bool(usable[0])
      cube_pos, cube_rot = cache.position[0].copy(), cache.rotation[0].copy()
      pos_err_mm = float(np.linalg.norm(cube_pos - nominal_pos) * 1000) if usable else np.nan
      in_place = usable and pos_err_mm < a.nominal_tol_mm
      ready_ticks = ready_ticks + 1 if in_place else 0
      ready = ready_ticks >= a.gate_ticks
      if usable:
        last_usable_time = now

      q, q_time = (None, -np.inf) if hand is None else hand.joints()
      tick_raw = tick_obs = None
      pressed, gui_keys = keys.read() + gui_keys, ""

      # ---------------- state machine ----------------
      if hand is not None:
        joints_fresh = q is not None and now - q_time < 0.2
        if "q" in pressed:
          if state == POLICY:
            enter_hold("quit")
          break
        if state == TELEOP:
          if a.driver_only:
            # The driver tracks its startup pose; in free space the measured pose is that target.
            cmd, cmd_time = (q, q_time) if q is not None else (None, -np.inf)
          else:
            cmd, cmd_time = hand.teleop_command()
          if cmd is not None and not a.driver_only:
            target = cmd
          if "h" in pressed:
            if cmd is None or now - cmd_time > 0.2 or not joints_fresh:
              message = "refused h: no fresh teleop command / joint state"
            else:
              # Hold the exact command the hand is already tracking. Like the sim warmup
              # target (default_joint_pos), it may sit outside the soft limits; only policy
              # actions are clamped to them.
              target = cmd.copy()
              try:
                if live:
                  hand.switch(True)
                  hand.send(target)
                enter_hold("current pose frozen, press o to open" if a.driver_only else "teleop frozen")
              except RuntimeError as e:
                message = f"switch to INFERENCE failed: {e}"
        elif state in (HOLD, BLEND):
          if "o" in pressed and state == HOLD:
            if q_open is None:
              message = f"refused o: {OPEN_POSE.name} missing (run compute_open_pose.py)"
            else:
              blend_from, blend_to, blend_s, blend_start, state = target.copy(), q_open, a.open_s, now, BLEND
              blend_kind = "OPEN"
              blend_done, message = "open: push the cube up into the hand, then g", "OPENING"
          if "g" in pressed and state == HOLD:
            blend_from, blend_to, blend_s, blend_start, state = target.copy(), q_default, a.close_s, now, BLEND
            blend_kind = "CLOSE"
            blend_done, message = "training grasp: let go of the cube, then p", "CLOSING to training grasp"
          if state == BLEND:
            s = min((now - blend_start) / blend_s, 1.0)
            s = s * s * (3 - 2 * s)
            target = blend_from + s * (blend_to - blend_from)
            if s >= 1.0:
              enter_hold(blend_done)
          if "t" in pressed and a.driver_only:
            message = "no teleop in --driver-only"
          elif "t" in pressed:
            if now - t_press < 2.0:
              try:
                if live:
                  hand.switch(False)
                state, message = TELEOP, "TELEOP (glove has control)"
              except RuntimeError as e:
                message = f"switch to TELEOP failed: {e}"
              print(f"\n{message}. Note: the controller re-sends its cached INFERENCE target for one "
                    "tick on the next 'h' switch.", flush=True)
            else:
              t_press, message = now, "press t again within 2 s to return to TELEOP"
          if "p" in pressed and state == HOLD:
            dev = np.abs(target - q_default)
            if not joints_fresh:
              message = "refused p: joint states stale"
            elif not (ready or a.force_start):
              message = "refused p: vision gate not READY"
            elif dev.max() > a.max_default_dev_rad:
              message = f"refused p: {JOINT_NAMES[int(dev.argmax())]} is {dev.max():.2f} rad from default"
            else:
              # Initialise exactly like a sim episode after warmup, but from the held grasp:
              # EMA state = current target, action history = equivalent action of that target.
              equivalent = np.clip((target - q_default) / scale, -1, 1)
              last, previous = equivalent.copy(), equivalent.copy()
              history = SensorHistory()
              goal = sample_goal(rng, cube_rot, a.max_goal_deg, a.min_goal_deg)
              goals_total += 1
              goal_hold = 0
              policy_start = goal_start = now
              state, message = POLICY, "POLICY"
              events.append((now, "POLICY start"))
              print(f"\n[{time.strftime('%H:%M:%S')}] POLICY start", flush=True)
        elif state == POLICY:
          if " " in pressed:
            enter_hold("operator")
          elif not joints_fresh:
            enter_hold("joint states stale")
          elif now - last_usable_time > a.max_blind_s:
            enter_hold("vision lost")
          elif usable and pos_err_mm > a.drop_tol_mm:
            enter_hold(f"cube {pos_err_mm:.0f} mm from nominal")

        if state == POLICY:
          if "n" in pressed or now - goal_start > a.goal_timeout_s:
            goal = sample_goal(rng, cube_rot, a.max_goal_deg, a.min_goal_deg)
            goals_total += 1
            goal_hold, goal_start = 0, now
          x = history.update(q, limits, target, previous, cube_pos, cube_rot, goal)
          if usable:
            with torch.inference_mode():
              raw = policy(torch.from_numpy(x)[None])[0].numpy().astype(np.float64)
          else:
            raw = last.copy()  # sim: hold the last action on missing/stale vision
          if not np.isfinite(raw).all():
            enter_hold("non-finite action")
          else:
            # The sim feeds the action from two steps back (scripts/eval_cube_wrist_student.py).
            previous, last = last, raw
            desired = np.clip(q_default + np.clip(raw, -1, 1) * scale, limits[:, 0], limits[:, 1])
            smoothed = alpha * desired + (1 - alpha) * target
            speed = a.startup_joint_speed if now - policy_start < a.startup_s else a.max_joint_speed
            target = target + np.clip(smoothed - target, -speed * ctrl_dt, speed * ctrl_dt)
            err = rotation_angle(cube_rot @ goal.T) if usable else np.nan
            goal_hold = goal_hold + 1 if usable and err < a.success_rad else 0
            if goal_hold == a.success_ticks:
              goals_reached += 1
              events.append((now, f"goal {goals_total} reached ({goals_reached} total)"))
              if a.stop_after_goals and goals_reached >= a.stop_after_goals:
                enter_hold(f"{goals_reached} goals reached")
              else:
                goal = sample_goal(rng, cube_rot, a.max_goal_deg, a.min_goal_deg)
                goals_total += 1
                goal_hold, goal_start = 0, now
            tick_raw, tick_obs = raw, x

        if state != TELEOP and target is not None:
          hand.send(target)

        if q is not None and target is not None:
          log["t"].append(now)
          log["state"].append(STATE_CODES[state])
          log["q"].append(q)
          log["target"].append(target.copy())
          log["pos"].append(cube_pos)
          log["rot"].append(cube_rot)
          log["fresh"].append(fresh)
          log["usable"].append(usable)
          log["goal"].append(np.eye(3) if goal is None else goal)
          log["raw"].append(np.full(20, np.nan) if tick_raw is None else tick_raw)
          log["obs"].append(np.full(207, np.nan, np.float32) if tick_obs is None else tick_obs)
          log["phase"].append(PHASE_CODES[blend_kind if state == BLEND else state])
          log["goal_index"].append(goals_total)
          log["goals_reached"].append(goals_reached)
          if recorder is not None:
            # One video frame per logged tick: the image whose detection this tick used.
            recorder.write(np.zeros((480, 640, 3), np.uint8) if rgb is None else rgb[:, :, ::-1])
          log["frame"].append(len(log["t"]) - 1 if recorder is not None else -1)
      elif "q" in pressed:
        break

      # ---------------- display ----------------
      if now - last_print > 0.5:
        last_print = now
        parts = [f"{state:6s}", f"vision {'OK' if usable else '--'}"]
        if pose is not None and new_pose is not None:
          parts.append(f"tags {len(pose.ids):2d} {pose.reprojection_error_px:.1f}px")
        parts.append(f"cube {pos_err_mm:4.1f}mm" if usable else "cube  n/a")
        if usable and state != POLICY:
          # Palm frame: +x away from the palm (down when palm-down), +y thumb/index side, +z fingertips.
          off = (T_tag @ cube_pos + TAG_IN_PALM_POS - nominal_palm) * 1000
          hint = [
            f"{label} {abs(v):.0f}"
            for v, label in zip(off, (("靠近掌心", "离开掌心"), ("往小指侧", "往拇指侧"), ("往手腕", "往指尖")))
            for label in [label[0] if v > 0 else label[1]]
            if abs(v) >= 3
          ]
          parts.append("挪动方块(mm): " + " ".join(hint) if hint else "位置到位")
        parts.append("READY" if ready else f"gate {ready_ticks}/{a.gate_ticks}")
        if target is not None:
          dev = np.abs(target - q_default)
          parts.append(f"dev {dev.max():.2f} {JOINT_NAMES[int(dev.argmax())][6:]}")
        if state == POLICY and usable:
          parts.append(f"goal err {np.degrees(rotation_angle(cube_rot @ goal.T)):5.1f}deg {goals_reached}/{goals_total}")
        if hand is not None and hand.error:
          parts.append("ROS: " + hand.error)
        if message:
          parts.append(message)
        print("\r" + " | ".join(parts) + "\033[K", end="", flush=True)

      if a.gui and rgb is not None:
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if usable:
          # Cube pose back in the optical camera frame.
          R_c = Rcam.T @ cube_rot
          t_c = Rcam.T @ (cube_pos - pcam)
          cv2.drawFrameAxes(img, camera.K, camera.dist, cv2.Rodrigues(R_c)[0], t_c, 0.035, 2)
          if goal is not None:
            cv2.drawFrameAxes(img, camera.K, camera.dist, cv2.Rodrigues(Rcam.T @ goal)[0], t_c, 0.02, 1)
        cv2.putText(img, f"{state} {'READY' if ready else ''} {message}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("wrist student handover", img)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
          gui_keys += chr(key)  # handled on the next tick, same as terminal keys
  except KeyboardInterrupt:
    if state == POLICY:
      enter_hold("Ctrl-C")
      if hand is not None and target is not None:
        hand.send(target)
  finally:
    keys.close()
    print("", flush=True)
    if recorder is not None:
      recorder.release()
    if log["t"]:
      np.savez_compressed(run_dir / "trajectory.npz", **{k: np.asarray(v) for k, v in log.items()})
      t0 = log["t"][0]
      (run_dir / "events.json").write_text(json.dumps([[round(t - t0, 3), e] for t, e in events], indent=1))
      print(f"Saved {len(log['t'])} ticks to {run_dir / 'trajectory.npz'}"
            + (f" and {run_dir / 'wrist.mp4'}" if recorder is not None else ""), flush=True)
      print(f"Goals reached: {goals_reached}. Render a demo video with:\n"
            f"  python deploy/wrist_student/render_handover_demo.py {run_dir}", flush=True)
    if live and hand is not None and hand.inference and a.driver_only:
      print("The driver keeps holding the last target. Remove the cube BEFORE stopping the driver:\n"
            "Ctrl-C on the driver disables the joints and the fingers go limp.", flush=True)
    elif live and hand is not None and hand.inference:
      print("Controller left in INFERENCE mode holding the last target. Remove the cube, then\n"
            "  ros2 service call /wuji_hand/right/switch_mode std_srvs/srv/SetBool '{data: false}'\n"
            "to give control back to the glove.", flush=True)
    camera.close()
    if hand is not None:
      hand.close()
    if a.gui:
      cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
