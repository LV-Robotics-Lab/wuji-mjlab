"""Build a relocatable engineering release with final models and calibration."""

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--models", type=Path, required=True)
  p.add_argument("--remote-root", required=True)
  a = p.parse_args()
  a.output.mkdir(parents=True, exist_ok=False)
  selections = [
    "src",
    "scripts",
    "deploy",
    "docs",
    "pyproject.toml",
    "pixi.toml",
    "pixi.lock",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CHANGELOG.md",
    "hardware/palm_down_cube_45mm",
    "hardware/wuji_hand_optical_mount",
    "hardware/franka_wuji_sim_assembly_20260930/franka_wuji_sim_assembly",
    "hardware/franka_wuji_sim_assembly_20260930/final_rgb640_assembly",
    "hardware/franka_wuji_sim_assembly_20260930/final_rgb640_demo",
    "hardware/franka_wuji_sim_assembly_20260930/fov_validation/right_arm_rgb_profiles.json",
    "hardware/franka_wuji_sim_assembly_20260930/fov_validation/remote_factory_intrinsics.json",
    "hardware/franka_wuji_sim_assembly_20260930/fov_validation/comparison.json",
    "hardware/franka_wuji_sim_assembly_20260930/fov_validation/eval_factory-vga",
    "hardware/franka_wuji_sim_assembly_20260930/hand_model_comparison.json",
    "hardware/franka_wuji_sim_assembly_20260930/official_hand1_provenance.json",
    "logs/rsl_rl/wuji_cube_wrist_student/camera_dagger_20260930_v2/student.pt",
    "logs/rsl_rl/wuji_cube_wrist_student/camera_dagger_20260930_v2/training.json",
    "logs/rsl_rl/wuji_cube_wrist_student/initial_bc_20260930/student.pt",
    "logs/rsl_rl/wuji_cube_downward/anypose_rewardfix_20260923_v1/rotate_full_so3_8000/final.pt",
  ]
  for relative in selections:
    source, dest = ROOT / relative, a.output / relative
    if not source.exists():
      raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
      shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".pytest_cache"),
      )
    else:
      shutil.copy2(source, dest)
  shutil.copytree(a.models, a.output / "release_models")
  # Relocate generated mesh paths and provenance references, including demo XMLs.
  for path in a.output.rglob("*"):
    if path.is_file() and path.suffix in {".xml", ".urdf", ".json", ".txt", ".py"}:
      content = path.read_text()
      if str(ROOT) in content:
        path.write_text(content.replace(str(ROOT), a.remote_root))
  instructions = f"""工控机工程版本：真实RGB640x480、法兰手侧、6度、无额外回退

工程目录：{a.remote_root}
独立版本目录，不覆盖工控机现有项目；包中不含SSH密码、虚拟环境、Git凭据或训练中间checkpoint。
包含训练/仿真源码、现有deploy接口、关键teacher/student、出厂标定、方块模型/贴纸、装配图片及完整demo。

先运行只读检查（不发送电机命令）：
cd {a.remote_root}
/home/gmlab/workspace/PrometheusV4/.venv/bin/python deploy/wrist_student/check_release.py --query-camera

CPU推理：release_models/policy.jit（Python torch.jit.load，207维输入/20维原始动作）。
ONNX：release_models/policy.onnx；归一化已内嵌，勿重复归一化。
动作映射、关节顺序/限位、相机变换：release_models/policy_config.json。
207维观测参考：src/wuji_mjlab/perception/cube_wrist.py 中 SensorHistory。
RGB序列号260322272638，640x480@30fps。factory原始畸变需要正确处理，不能把原图当成无畸变针孔画面。

仿真/训练源码要求Python>=3.11及pixi.toml对应依赖；现有Prometheus Python3.10只用于独立推理/相机检查。
当前工控机缺少wujihandpy/wuji_sdk和AprilTag检测依赖，需要独立配置部署环境。
遥操作预抓握 -> 保持 -> 视觉有效 -> 平滑策略接管的电机桥接尚未完成；本检查不表示已可驱动真机。
遥操作接管需用实测关节、已发送目标、历史动作初始化，不能直接执行默认仿真reset/home。

演示：hardware/franka_wuji_sim_assembly_20260930/final_rgb640_demo/official_dual_view.mp4
装配：hardware/franka_wuji_sim_assembly_20260930/final_rgb640_assembly/assembly.png
仿真基准：真实VGA内参下58/64目标成功，15/16存活；不是实机成功率。
"""
  (a.output / "DEPLOY_ON_GMLAB.zh-CN.txt").write_text(instructions)
  files = [
    {
      "path": str(path.relative_to(a.output)),
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "bytes": path.stat().st_size,
    }
    for path in sorted(a.output.rglob("*"))
    if path.is_file()
  ]
  manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "remote_root": a.remote_root,
    "source_root": str(ROOT),
    "files": files,
    "motor_control_enabled": False,
  }
  (a.output / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
  archive = a.output.with_suffix(".tar.gz")
  with tarfile.open(archive, "w:gz") as tar:
    for path in sorted(a.output.rglob("*")):
      if path.is_file():
        tar.add(path, arcname=str(path.relative_to(a.output)), recursive=False)
  print(
    json.dumps(
      {
        "archive": str(archive),
        "file_count": len(files),
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
