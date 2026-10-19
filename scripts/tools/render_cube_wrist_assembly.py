"""Render the current palm-down flange, hand, and wrist camera assembly."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from train.train_cube_downward import make_cfg  # noqa: E402


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  cfg = make_cfg(
    "rotate", 1, 7400, cube_edge_m=0.045, cube_mass_kg=0.028192, wrist_assembly=True
  )
  xml = (
    ROOT
    / "hardware/franka_wuji_sim_assembly_20260930/fov_validation/eval_factory-vga/tagged_cube.xml"
  )
  cfg.scene.entities["object"].spec_fn = partial(mujoco.MjSpec.from_file, str(xml))
  cfg.viewer.width, cfg.viewer.height = 800, 640
  cfg.commands["reorient_command"].debug_vis = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
  renderer = None
  try:
    env.reset()
    model = env.sim.mj_model
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot/d405_b")
    env._offline_renderer.update(env.sim.data)
    data = env._offline_renderer._data
    position = data.cam_xpos[cid].copy()
    axis = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
    pitch = float(np.degrees(np.arctan2(axis[2], np.linalg.norm(axis[:2]))))
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = 0.255
    renderer = mujoco.Renderer(model, height=640, width=800)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    panels = []
    for name, azimuth, elevation in [
      ("SIDE VIEW", 90, 0),
      ("ASSEMBLY OVERVIEW", 125, -20),
    ]:
      camera = mujoco.MjvCamera()
      mujoco.mjv_defaultCamera(camera)
      camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, 0.5
      camera.lookat = [0.047, 0, 0.49]
      renderer.update_scene(data, camera=camera, scene_option=option)
      panel = Image.fromarray(renderer.render().copy())
      draw = ImageDraw.Draw(panel)
      draw.rectangle((0, 0, 800, 78), fill=(12, 20, 28))
      draw.text((20, 12), f"{name} | PALM DOWN | WUJI HAND 1", fill="white", font=font)
      draw.text(
        (20, 42),
        "D405 on flange HAND SIDE | 6 deg pitch | no extra backoff",
        fill=(190, 210, 230),
        font=font,
      )
      if azimuth == 90:

        def pixel(p):
          return (
            int(400 + (p[0] - 0.047) / 0.255 * 640),
            int(320 - (p[2] - 0.49) / 0.255 * 640),
          )

        start, end = pixel(position), pixel(position + axis * 0.15)
        draw.line((start, end), fill=(255, 70, 60), width=4)
        draw.ellipse(
          (start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5), fill=(255, 70, 60)
        )
        draw.text((start[0] - 35, start[1] + 34), "D405 lens", fill="white", font=font)
        draw.text(
          (start[0] + 95, start[1] + 34),
          "red = optical axis",
          fill=(255, 110, 100),
          font=font,
        )
      draw.text(
        (20, 572),
        "RGB 640x480 | factory FOV H78.26 / V62.89 deg",
        fill="white",
        font=font,
      )
      draw.text(
        (20, 600),
        "Cube 45 mm / 28.192 g | mounting based on provided assembly",
        fill="white",
        font=font,
      )
      panel.save(args.output / ("side.png" if azimuth == 90 else "overview.png"))
      panels.append(panel)
    combined = Image.new("RGB", (1600, 640))
    combined.paste(panels[0], (0, 0))
    combined.paste(panels[1], (800, 0))
    combined.save(args.output / "assembly.png")
    (args.output / "geometry.json").write_text(
      json.dumps(
        {
          "camera_world_position_m": position.tolist(),
          "optical_axis": axis.tolist(),
          "pitch_deg": pitch,
          "camera_mount": json.loads(
            (
              ROOT
              / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/source.json"
            ).read_text()
          ),
        },
        indent=2,
      )
    )
  finally:
    if renderer:
      renderer.close()
    env.close()


if __name__ == "__main__":
  main()
