"""Create a venv and install the exact Qwen3.5/DoRA runtime from wheels.

The code is environment-agnostic: pass any local wheel directory. Kaggle is only
one possible place where that directory may be mounted.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import subprocess
from pathlib import Path


TRANSFORMERS_VERSION = "5.14.1"
SAFETENSORS_VERSION = "0.8.0"
PEFT_VERSION = "0.19.1"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel-dir", help="optional local wheel directory; package index is fallback")
    ap.add_argument("--venv", default=".venv")
    ap.add_argument(
        "--python",
        default="3.12",
        help="Python interpreter for the venv; the project requires Python 3.12",
    )
    ap.add_argument(
        "--system-site-packages", action="store_true",
        help="reuse preinstalled Torch/CUDA packages (useful on managed GPU images)",
    )
    args = ap.parse_args()
    wheels = Path(args.wheel_dir).resolve() if args.wheel_dir else None
    if wheels is not None and (not wheels.is_dir() or not any(wheels.rglob("*.whl"))):
        print(f"Local wheels unavailable at {wheels}; using package index", flush=True)
        wheels = None
    venv = Path(args.venv).resolve()
    create = ["uv", "venv", str(venv), "--python", str(args.python)]
    if args.system_site_packages:
        create.append("--system-site-packages")
    if not (venv / "pyvenv.cfg").is_file():
        run(create)
    venv_python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    version = subprocess.check_output(
        [str(venv_python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True,
    ).strip()
    if version != "3.12":
        raise SystemExit(
            f"Incompatible existing venv uses Python {version}; remove {venv} and recreate it with Python 3.12"
        )

    install = [
        "uv", "pip", "install", "--python", str(venv_python),
        f"transformers=={TRANSFORMERS_VERSION}",
        f"safetensors=={SAFETENSORS_VERSION}",
        "accelerate", "sentencepiece",
    ]
    if wheels is not None:
        install[5:5] = ["--find-links", str(wheels)]
    run(install)

    # Reuse an exact system-site PEFT when available. This avoids unnecessary
    # network access on managed offline images such as Kaggle.
    probe = subprocess.run(
        [str(venv_python), "-c", "import importlib.metadata as m; print(m.version('peft'))"],
        text=True, capture_output=True,
    )
    installed_peft = probe.stdout.strip() if probe.returncode == 0 else None

    # Search only inside the explicitly supplied wheel dataset. Never scan all
    # of /kaggle/input (or any other parent mount).
    peft_wheels = sorted(wheels.rglob(f"peft-{PEFT_VERSION}-*.whl")) if wheels else []
    if installed_peft == PEFT_VERSION:
        print(f"Reusing accessible peft=={PEFT_VERSION}", flush=True)
        peft_command = None
    elif peft_wheels:
        peft_command = [
            "uv", "pip", "install", "--python", str(venv_python),
            "--no-index", "--no-deps", "--reinstall", str(peft_wheels[0]),
        ]
    else:
        # This is the original trainer behaviour: PEFT may be absent from the
        # runtime-wheel dataset, in which case install the exact pin from the
        # configured package index.
        peft_command = [
            "uv", "pip", "install", "--python", str(venv_python),
            "--no-deps", "--reinstall", f"peft=={PEFT_VERSION}",
        ]
    if peft_command is not None:
        run(peft_command)
    importlib.invalidate_caches()

    run([str(venv_python), "-c", (
        "import importlib.metadata as m; "
        "print({p:m.version(p) for p in ('transformers','safetensors','peft')})"
    )])
    print(f"VENV_READY={venv}", flush=True)
    print(f"Run the solution with: {venv_python} solution.py --help", flush=True)


if __name__ == "__main__":
    main()
