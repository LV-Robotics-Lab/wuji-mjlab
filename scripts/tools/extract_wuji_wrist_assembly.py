"""Extract the rigid wrist rig while keeping the canonical training hand."""

import argparse
import copy
import json
import shutil
import xml.etree.ElementTree as E
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
  "--source",
  type=Path,
  default=root / "hardware/franka_wuji_sim_assembly_20260930/franka_wuji_sim_assembly",
)
parser.add_argument(
  "--camera-backoff-mm",
  type=float,
  default=0.0,
  help="Optional optical backoff from the source bracket; default 0 keeps the camera on the flange hand side",
)
parser.add_argument(
  "--camera-pitch-deg",
  type=float,
  default=6.0,
  help="Optical elevation in fixed palm-down world: 0 = horizontal, positive = upward",
)
parser.add_argument(
  "--camera-calibration",
  type=Path,
  default=root
  / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json",
  help="RGB factory calibration matching the rendered resolution",
)
args = parser.parse_args()
if args.camera_backoff_mm < 0:
  parser.error("camera-backoff-mm must be non-negative")
src = args.source.resolve()
dest = root / "src/wuji_mjlab/assets/robots/wuji_hand"
scene = E.parse(src / "scene.xml").getroot()
connector = copy.deepcopy(scene.find(".//body[@name='right_connector_link']"))
dock = copy.deepcopy(connector.find("body[@name='right_hand_docking_link']"))
palm = dock.find("body[@name='right_palm_link']")
# Connector -> palm transform, inverted to attach the fixture to the trained palm root.
q = np.array(list(map(float, dock.get("quat").split())))
R = Rotation.from_quat(np.roll(q, -1)).as_matrix()
t = np.array(list(map(float, dock.get("pos").split()))) + R @ np.array([0, 0, 0.013])
Ri = R.T
ti = -Ri @ t
connector.remove(connector.find("body[@name='right_hand_docking_link']"))
connector.set("pos", " ".join(map(str, ti)))
connector.set(
  "quat", " ".join(map(str, np.roll(Rotation.from_matrix(Ri).as_quat(), 1)))
)
dock.remove(palm)
dock.set("pos", "0 0 -0.013")
dock.set("quat", "1 0 0 0")
camera = connector.find("body[@name='d405_b_camera_link']")
original_camera_pos = np.array(list(map(float, camera.get("pos").split())))
camera_quat = np.array(list(map(float, camera.get("quat").split())))
camera_rotation = Rotation.from_quat(np.roll(camera_quat, -1)).as_matrix()
# Fixed palm-down task has palm R_y(+90). World camera right=-Y, up=+Z.
palm_world_rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
horizontal_world_camera = np.array(
  [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)
world_camera = (
  Rotation.from_euler("y", -args.camera_pitch_deg, degrees=True).as_matrix()
  @ horizontal_world_camera
)
camera_rotation = Ri.T @ palm_world_rotation.T @ world_camera
camera.set(
  "quat",
  " ".join(map(str, np.roll(Rotation.from_matrix(camera_rotation).as_quat(), 1))),
)
camera_delta = camera_rotation[:, 2] * (args.camera_backoff_mm / 1000.0)
camera.set("pos", " ".join(map(str, original_camera_pos + camera_delta)))
camera_intrinsics_source = "Source assembly (historical calibration)"
if args.camera_calibration.is_file():
  intrinsics = json.loads(args.camera_calibration.read_text())
  w, h = intrinsics["width"], intrinsics["height"]
  lens = camera.find("camera")
  lens.set("resolution", f"{w} {h}")
  lens.set("focalpixel", f"{intrinsics['fx']} {intrinsics['fy']}")
  lens.set(
    "principalpixel",
    f"{(w - 1) / 2 - intrinsics['cx']} {(h - 1) / 2 - intrinsics['cy']}",
  )
  camera_intrinsics_source = str(args.camera_calibration.resolve())
wrapper = E.Element("mujoco", model="wuji_wrist_fixture")
asset = E.SubElement(wrapper, "asset")
body = E.SubElement(wrapper, "worldbody")
body.append(connector)
body.append(dock)
meshdir = dest / "meshes/wrist_assembly"
meshdir.mkdir(exist_ok=True)
for meshname in sorted(
  {g.get("mesh") for g in wrapper.findall(".//geom") if g.get("mesh")}
):
  mesh = copy.deepcopy(scene.find(f"asset/mesh[@name='{meshname}']"))
  source = src / mesh.get("file")
  shutil.copy2(source, meshdir / source.name)
  mesh.set("file", source.name)
  mesh.set("name", "wrist_" + meshname)
  asset.append(mesh)
  for g in wrapper.findall(".//geom"):
    if g.get("mesh") == meshname:
      g.set("mesh", "wrist_" + meshname)
contact = E.SubElement(wrapper, "contact")
for a, b in [
  ("right_connector_link", "d405_b_camera_link"),
  ("right_hand_docking_link", "right_palm_link"),
  ("right_connector_link", "right_hand_docking_link"),
]:
  E.SubElement(contact, "exclude", body1=a, body2=b)
E.indent(wrapper)
E.ElementTree(wrapper).write(dest / "mjcf/wrist_assembly.xml", encoding="unicode")
(dest / "meshes/wrist_assembly/source.json").write_text(
  json.dumps(
    {
      "source_scene": str(src / "scene.xml"),
      "source_palm_to_connector_pos_m": ti.tolist(),
      "source_palm_to_connector_rotation": Ri.tolist(),
      "hand_model": "Use canonical trained hand, not source assembly hand (different inertias).",
      "camera": "D405 intrinsics preserved; optical pitch set for the fixed palm-down mount and backed off along optical axis.",
      "camera_mount_reference": "Source assembly hand-side bracket; no additional backoff by default",
      "camera_intrinsics_source": camera_intrinsics_source,
      "camera_pitch_deg": args.camera_pitch_deg,
      "pitch_reference": "Optical elevation in the fixed R_y(+90) palm-down world",
      "camera_backoff_mm": args.camera_backoff_mm,
      "original_camera_pos_connector_m": original_camera_pos.tolist(),
      "camera_delta_connector_m": camera_delta.tolist(),
      "camera_pos_connector_m": (original_camera_pos + camera_delta).tolist(),
    },
    indent=2,
  )
  + "\n"
)
print("Extracted connector, docking and D405 with two meshes")
