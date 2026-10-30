#!/usr/bin/env python3
"""
End-to-end headless mocap pipeline.

Accepts one take with multiple camera videos and runs:
  1. extract_keypoints --calib  (board)
  2. extract_keypoints          (pose)
  3. calibrate_board
  4. reconstruct_take
Optionally:
  5. build_rig.py
  6. retarget.py

Run with --dry-run to print the commands without executing them.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

def cmd(*parts):
    return " ".join(str(p) for p in parts)

def run_cmd(command, dry_run):
    print(">>>", command)
    if dry_run:
        return 0
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode

def main():
    p = argparse.ArgumentParser(description="Batch mocap pipeline")
    p.add_argument("--take", help="Take name, required unless --takes-dir used")
    p.add_argument("--videos", nargs="+", help="Video files per camera in order")
    p.add_argument("--takes-dir", help="Directory with subfolders per take, each containing cam*.mp4")
    p.add_argument("--cam-ids", required=True, help="Comma separated camera ids")
    p.add_argument("--workdir", default="work", help="Working directory")
    p.add_argument("--kp-dir", default=None, help="Keypoints output dir")
    p.add_argument("--mocap-dir", default=None, help="Mocap output dir")
    p.add_argument("--board", action="store_true", help="Board calibration enabled")
    p.add_argument("--board-squares-x", type=int, default=7)
    p.add_argument("--board-squares-y", type=int, default=5)
    p.add_argument("--square-len", type=float, default=0.10)
    p.add_argument("--marker-len", type=float, default=0.075)
    p.add_argument("--board-start", type=float, default=0)
    p.add_argument("--board-end", type=float, default=20)
    p.add_argument("--pose-start", type=float, default=20)
    p.add_argument("--model", default="heavy", choices=["lite","full","heavy"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--build-rig", action="store_true")
    p.add_argument("--rig-preset", default="default", help="Rigify preset name")
    p.add_argument("--rig-height", type=float, default=1.75)
    p.add_argument("--leg-scale", type=float, default=1.08)
    p.add_argument("--arm-scale", type=float, default=0.95)
    p.add_argument("--rig-name", default="rig")
    p.add_argument("--rig-out", default=None)
    p.add_argument("--retarget-out", default=None)
    p.add_argument("--control-mode", default="FK", choices=["FK","LEGS_IK_ARMS_FK","BOTH"])
    p.add_argument("--embed", action="store_true", help="Embed npz and add-on in blend")
    p.add_argument("--blender", default="/opt/blender/blender")
    p.add_argument("--mr", default="/home/llama/synclip/blender/mocap_retarget")
    args = p.parse_args()

    if not args.takes_dir and not args.take:
        p.error("--take required unless --takes-dir is used")

    work = Path(args.workdir)
    kp_dir = Path(args.kp_dir or work / "kp")
    mocap_dir = Path(args.mocap_dir or work / "mocap")
    kp_dir.mkdir(parents=True, exist_ok=True)
    mocap_dir.mkdir(parents=True, exist_ok=True)

    # Support takes-dir mode
    takes = []
    if args.takes_dir:
        base = Path(args.takes_dir)
        for sub in sorted(base.iterdir()):
            if sub.is_dir():
                vids = sorted([p for p in sub.glob("*.mp4") if p.is_file()])
                if vids:
                    takes.append((sub.name, vids))
        if not takes:
            sys.exit("No takes found")
    else:
        if not args.videos:
            sys.exit("--videos required when --takes-dir not used")
        takes.append((args.take, [Path(v) for v in args.videos]))

    for take_name, videos in takes:
        print(f"\n=== Take {take_name} ===")
        if len(videos) != len(args.cam_ids.split(",")):
            sys.exit(f"Video count {len(videos)} != cam-ids count")

        # 1. Extract board keypoints
        if args.board:
            cmd_extract_calib = cmd(
                "python -m capmo.tools.extract_keypoints",
                "--calib",
                f"--start {args.board_start}",
                f"--end {args.board_end}",
                "--calib-every 1",
                f"--squares-x {args.board_squares_x}",
                f"--squares-y {args.board_squares_y}",
                f"--square-len {args.square_len}",
                f"--marker-len {args.marker_len}",
                f"--out {kp_dir}",
                f"--take {take_name}",
                f"--cam-ids {args.cam_ids}",
                " ".join(str(v) for v in videos)
            )
            run_cmd(cmd_extract_calib, args.dry_run)

        # 2. Extract pose keypoints
        cmd_extract_pose = cmd(
            "python -m capmo.tools.extract_keypoints",
            f"--start {args.pose_start}",
            f"--out {kp_dir}",
            f"--take {take_name}",
            f"--model {args.model}",
            f"--cam-ids {args.cam_ids}",
            "--keyframes 6",
            "--proxy",
            " ".join(str(v) for v in videos)
        )
        run_cmd(cmd_extract_pose, args.dry_run)

        # 3. Calibrate board
        if args.board:
            rig_json = kp_dir / take_name / "rig.json"
            cmd_calib = cmd(
                "python -m capmo.tools.calibrate_board",
                str(kp_dir / take_name),
                f"--out {rig_json}",
                f"--report {kp_dir / take_name / 'rig_report.json'}",
                f"--overlay-dir {kp_dir / take_name / 'rig_check'}"
            )
            run_cmd(cmd_calib, args.dry_run)

            # 4. Reconstruct with rig
            cmd_recon = cmd(
                "python -m capmo.tools.reconstruct_take",
                str(kp_dir / take_name),
                f"--out {mocap_dir}",
                f"--rig {rig_json}",
                f"--start {args.pose_start}"
            )
            run_cmd(cmd_recon, args.dry_run)
        else:
            # Self-calibration
            cmd_recon = cmd(
                "python -m capmo.tools.reconstruct_take",
                str(kp_dir / take_name),
                f"--out {mocap_dir}"
            )
            run_cmd(cmd_recon, args.dry_run)

        # Optional rig + retarget
        if args.build_rig:
            rig_blend = Path(args.rig_out) if args.rig_out else work / f"{take_name}_rig.blend"
            retarget_blend = Path(args.retarget_out) if args.retarget_out else work / f"{take_name}_retargeted.blend"
            npz_path = mocap_dir / f"{take_name}_joints3d.npz"
            cmd_build = cmd(
                args.blender, "-b", "--factory-startup",
                "--python", f"{args.mr}/build_rig.py",
                "--",
                f"--out {rig_blend}",
                f"--preset {args.rig_preset}",
                f"--height {args.rig_height}",
                f"--leg-scale {args.leg_scale}",
                f"--arm-scale {args.arm_scale}",
                f"--rig-name {args.rig_name}"
            )
            run_cmd(cmd_build, args.dry_run)

            set_args = f"--set control_mode={args.control_mode}"
            embed_flag = "--embed" if args.embed else ""
            cmd_retarget = cmd(
                args.blender, "-b", "--factory-startup",
                "--python", f"{args.mr}/retarget.py",
                "--",
                f"--blend {rig_blend}",
                f"--npz {npz_path}",
                f"--out {retarget_blend}",
                set_args,
                embed_flag
            )
            run_cmd(cmd_retarget, args.dry_run)

if __name__ == "__main__":
    main()
