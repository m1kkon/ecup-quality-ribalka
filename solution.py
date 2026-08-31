#!/usr/bin/env python3
"""Single entry point for training and packaging the E-CUP solution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LVJ_CATEGORY = "Легковоспламеняющиеся"
DEFAULT_DATA = ROOT / "data" / "data.csv"
DEFAULT_QWEN = "Qwen/Qwen3.5-4B"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def train_bad(args: argparse.Namespace) -> None:
    data = Path(args.data).resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Training data not found: {data}")
    spec = ROOT / "bad" / "specs" / "spec_es5_vote3.json"
    run([
        sys.executable, "-m", "src.fit_final", "--spec", str(spec),
        "--data", str(data), "--out", str(Path(args.out).resolve()),
    ], cwd=ROOT / "bad")


def train_lvj(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    if args.data:
        env["LVJ_TRAIN_DATA_PATH"] = str(Path(args.data).resolve())
    if args.model:
        model = Path(args.model).expanduser()
        env["LVJ_BASE_MODEL_PATH"] = str(model.resolve()) if model.exists() else args.model
    if args.out:
        env["LVJ_OUTPUT_ROOT"] = str(Path(args.out).resolve())
    print("+", sys.executable, ROOT / "lvj" / "train_dora.py", flush=True)
    subprocess.run([sys.executable, str(ROOT / "lvj" / "train_dora.py")], env=env, check=True)


def exact_key(name: object, description: object) -> str:
    payload = json.dumps([str(name), str(description)], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_exact_lookup(data_path: Path, output: Path) -> dict:
    import pandas as pd

    frame = pd.read_csv(data_path, keep_default_na=False)
    required = {"name", "description", "category", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training data misses columns: {sorted(missing)}")
    lvj = frame.loc[frame.category.astype(str).eq(LVJ_CATEGORY)].copy()
    grouped: dict[str, set[int]] = {}
    for row in lvj.itertuples(index=False):
        grouped.setdefault(exact_key(row.name, row.description), set()).add(int(row.label))
    conflicts = {key: labels for key, labels in grouped.items() if len(labels) != 1}
    entries = {key: next(iter(labels)) for key, labels in grouped.items() if len(labels) == 1}
    payload = {
        "schema_version": 1,
        "category": LVJ_CATEGORY,
        "key_contract": "sha256(UTF-8 compact JSON array [exact name, exact description])",
        "source_sha256": sha256(data_path),
        "source_lvj_rows": int(len(lvj)),
        "exact_unique_cards": int(len(grouped)),
        "duplicate_rows": int(len(lvj) - len(grouped)),
        "conflicting_cards_excluded": int(len(conflicts)),
        "entries": dict(sorted(entries.items())),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return payload


def find_training_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "processor").is_dir() and (path / "checkpoints").is_dir():
        return path
    matches = [p.parent for p in path.rglob("processor/processor_config.json")
               if (p.parent.parent / "checkpoints").is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one LVJ training root under {path}, found {matches}")
    return matches[0]


def copy_peft_vendor(destination: Path, explicit: str | None) -> None:
    source = Path(explicit).resolve() if explicit else None
    if source is None:
        spec = importlib.util.find_spec("peft")
        if spec is None or spec.origin is None:
            raise FileNotFoundError("PEFT is not installed; pass --peft-source /path/to/site-packages")
        source = Path(spec.origin).parent.parent
    package = source / "peft" if (source / "peft").is_dir() else source
    if package.name != "peft" or not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"PEFT package not found under {source}")
    shutil.copytree(package, destination / "peft")
    parent = package.parent
    dist_infos = sorted(parent.glob("peft-*.dist-info"))
    if dist_infos:
        shutil.copytree(dist_infos[-1], destination / dist_infos[-1].name)


def patch_runtime_contract(run_path: Path, adapter_sha: str, lookup_sha: str,
                           lookup_entries: int, rules_sha: str) -> None:
    text = run_path.read_text(encoding="utf-8")
    replacements = {
        "EXPECTED_ADAPTER_SHA256": f'"{adapter_sha}"',
        "EXPECTED_EXACT_LOOKUP_SHA256": f'"{lookup_sha}"',
        "EXPECTED_EXACT_LOOKUP_ENTRIES": str(lookup_entries),
        "EXPECTED_ORDER_AWARE_RULES_SHA256": f'"{rules_sha}"',
    }
    for name, value in replacements.items():
        text, count = re.subn(rf"^{name}\s*=\s*.+$", f"{name} = {value}", text,
                              count=1, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"Could not patch runtime constant {name}")
    run_path.write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    template = ROOT / "submission" / "template"
    for name in ("run.py", "lvj_prompting.py", "lvj_order_aware_rules.py"):
        shutil.copy2(template / name, out / name)

    bad_artifacts = Path(args.bad_artifacts).resolve()
    if not (bad_artifacts / "ensemble.json").is_file():
        raise FileNotFoundError(f"BAD ensemble.json not found: {bad_artifacts}")
    shutil.copytree(bad_artifacts, out / "bad_artifacts")
    shutil.copytree(ROOT / "bad" / "inference", out / "bad_src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    training_root = find_training_root(Path(args.lvj_training))
    checkpoint_name = f"epoch_{float(args.epoch):05.3f}".replace(".", "p")
    adapter = training_root / "checkpoints" / checkpoint_name / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LVJ adapter not found: {adapter}")
    shutil.copytree(adapter, out / "model" / "adapter")
    shutil.copytree(training_root / "processor", out / "model" / "processor")

    lookup = out / "lvj_train_exact_lookup.json"
    lookup_payload = build_exact_lookup(Path(args.data).resolve(), lookup)
    if lookup_payload["conflicting_cards_excluded"]:
        raise RuntimeError("Exact lookup contains conflicting labels")

    vendor = out / "vendor"
    vendor.mkdir()
    copy_peft_vendor(vendor, args.peft_source)
    adapter_sha = sha256(out / "model" / "adapter" / "adapter_model.safetensors")
    rules_sha = sha256(out / "lvj_order_aware_rules.py")
    patch_runtime_contract(out / "run.py", adapter_sha, sha256(lookup),
                           len(lookup_payload["entries"]), rules_sha)

    manifest = {
        "entrypoint": "run.py",
        "lvj_routing": "exact lookup -> best.zip order-aware rules -> DoRA fallback",
        "lvj_epoch": float(args.epoch),
        "lvj_adapter_sha256": adapter_sha,
        "lvj_rules_sha256": rules_sha,
        "exact_lookup_sha256": sha256(lookup),
        "exact_lookup_entries": len(lookup_payload["entries"]),
        "bad_ensemble": json.loads((out / "bad_artifacts" / "ensemble.json").read_text()),
    }
    (out / "solution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metadata.json").write_text(
        json.dumps({"title": "E-CUP quality reproducible"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    if args.zip:
        archive = Path(shutil.make_archive(str(out), "zip", root_dir=out))
        print(f"SUBMISSION_ZIP={archive} sha256={sha256(archive)}", flush=True)
    print(f"SUBMISSION_DIR={out}", flush=True)


def freeze_env(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    # `uv pip freeze` intentionally ignores packages inherited through
    # --system-site-packages. Prefer pip from the active interpreter so a
    # managed Kaggle image is captured in full; retain uv as a fallback for a
    # minimal uv venv that was created without pip.
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        result = subprocess.run(
            ["uv", "pip", "freeze", "--python", sys.executable],
            check=True, capture_output=True, text=True,
        )
    lines = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ENV_LOCK={output} packages={len(lines)}", flush=True)


def all_pipeline(args: argparse.Namespace) -> None:
    work = Path(args.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    bad_out = work / "bad_artifacts"
    lvj_out = work / "lvj_training"
    submission_out = work / "submission"
    train_bad(argparse.Namespace(data=args.data, out=str(bad_out)))
    train_lvj(argparse.Namespace(data=args.data, model=args.model, out=str(lvj_out)))
    build(argparse.Namespace(
        data=args.data,
        bad_artifacts=str(bad_out),
        lvj_training=str(lvj_out),
        epoch=args.epoch,
        peft_source=args.peft_source,
        out=str(submission_out),
        zip=args.zip,
    ))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train-bad")
    p.add_argument("--data", default=str(DEFAULT_DATA)); p.add_argument("--out", required=True)
    p.set_defaults(func=train_bad)
    p = sub.add_parser("train-lvj")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model", default=DEFAULT_QWEN, help="local directory or Hugging Face model ID")
    p.add_argument("--out")
    p.set_defaults(func=train_lvj)
    p = sub.add_parser("build")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--bad-artifacts", required=True)
    p.add_argument("--lvj-training", required=True)
    p.add_argument("--epoch", type=float, default=2.0)
    p.add_argument("--peft-source")
    p.add_argument("--out", required=True)
    p.add_argument("--zip", action="store_true")
    p.set_defaults(func=build)
    p = sub.add_parser("freeze-env")
    p.add_argument("--output", default="requirements-kaggle-frozen.txt")
    p.set_defaults(func=freeze_env)
    p = sub.add_parser("all", help="train BAD + LVJ and build the submission")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model", default=DEFAULT_QWEN, help="local directory or Hugging Face model ID")
    p.add_argument("--workdir", required=True)
    p.add_argument("--epoch", type=float, default=2.0)
    p.add_argument("--peft-source")
    p.add_argument("--zip", action="store_true")
    p.set_defaults(func=all_pipeline)
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
