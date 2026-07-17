#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from Pelagia.cli.environments import doctor_profiles, sync_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and inspect Pelagia worker environments.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    sync = subcommands.add_parser("sync", help="Create and install a named environment.")
    sync.add_argument("profile", choices=["cpu", "dev", "ml-metal", "ml-cuda"])
    sync.add_argument("--root", type=Path, default=ROOT_DIR)
    sync.add_argument("--python", help="Python version request or interpreter path (default: 3.12).")
    sync.add_argument("--uv", type=Path, help="Path to the uv executable.")
    sync.add_argument("--imagecodecs-wheel", type=Path)
    sync.add_argument("--dry-run", action="store_true")

    doctor = subcommands.add_parser("doctor", help="Report whether named environments are ready.")
    doctor.add_argument("profile", nargs="?", default="all", choices=["all", "cpu", "gpu-ml"])
    doctor.add_argument("--root", type=Path, default=ROOT_DIR)
    doctor.add_argument("--require-gpu", action="store_true")
    doctor.add_argument("--require-jpegxs", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "sync":
            result = sync_profile(
                args.profile,
                root=args.root,
                python=args.python,
                uv=args.uv,
                imagecodecs_wheel=args.imagecodecs_wheel,
                dry_run=args.dry_run,
            )
            status = 0
        else:
            result = doctor_profiles(
                args.profile,
                root=args.root,
                require_gpu=args.require_gpu,
                require_jpegxs=args.require_jpegxs,
            )
            status = 0 if result["healthy"] else 1
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
