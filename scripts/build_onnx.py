#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE_COPY = Path.home() / ".cache" / "manafold" / "onnxruntime-source"


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  workspace = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", Path.cwd()))
  declared_source = resolve_input(args.onnxruntime_source, workspace)
  if declared_source.is_file():
    declared_source = declared_source.parent
  source = prepare_source_copy(
    declared_source,
    resolve_path(args.source_copy_dir, workspace),
    source_id=args.source_id,
  )
  output_dir = resolve_path(args.output_dir, workspace)
  worker_dir = None if args.no_install_worker else resolve_path(args.worker_dir, workspace)
  ops_config = resolve_path(args.ops_config, workspace)

  version = read_ort_version(source)
  build_python = Path(sys.executable).resolve()
  ninja = resolve_input(args.ninja_executable, workspace)
  if ninja.is_dir():
    ninja = ninja / "bin" / "ninja"
  if not ninja.is_file() or not os.access(ninja, os.X_OK):
    raise SystemExit(f"Bazel's declared Ninja executable is missing: {ninja}")
  patch = resolve_input(args.patch_executable, workspace)
  if not patch.is_file() or not os.access(patch, os.X_OK):
    raise SystemExit(f"Bazel's declared GNU patch executable is missing: {patch}")
  npm = resolve_executable(args.npm, option="--npm")

  build_root = resolve_path(
    args.build_dir or source / "build" / "manafold-reduced",
    workspace,
  )
  build_runtime(
    source=source,
    build_root=build_root,
    ops_config=ops_config,
    build_python=build_python,
    ninja=ninja,
    patch=patch,
    jobs=args.jobs,
  )
  artifacts = package_web_runtime(
    source=source,
    build_root=build_root,
    output_dir=output_dir,
    version=version,
    npm=npm,
    install_js_dependencies=not args.skip_js_install,
  )
  if worker_dir is not None:
    install_worker_runtime(artifacts, worker_dir)

  print(json.dumps({
    "status": "completed",
    "onnxruntime_version": version,
    "operator_config": str(ops_config),
    "output_dir": str(output_dir),
    "worker_dir": str(worker_dir) if worker_dir else None,
    "files": {
      name: file_metadata(path) for name, path in artifacts.items()
    },
  }, indent=2, sort_keys=True))
  return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Build the operator-reduced ONNX Runtime Web package used by the "
      "Manafold Cloudflare Worker."
    )
  )
  parser.add_argument(
    "--onnxruntime-source",
    type=Path,
    required=True,
    help="Bazel-declared recursive ONNX Runtime source tree or VERSION_NUMBER file.",
  )
  parser.add_argument(
    "--source-id",
    required=True,
    help="Immutable source revision declared by the Bazel target.",
  )
  parser.add_argument(
    "--source-copy-dir",
    type=Path,
    default=DEFAULT_SOURCE_COPY,
    help="Writable cached copy of the Bazel-declared source tree.",
  )
  parser.add_argument(
    "--ops-config",
    type=Path,
    default=Path("src/onnx-runtime/runtime/required_operators.config"),
    help="Reduced-operator configuration shared by deployable set-model artifacts.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("src/onnx-runtime/runtime/generated"),
    help="Directory for the reduced runtime pair and build manifest.",
  )
  parser.add_argument(
    "--worker-dir",
    type=Path,
    default=Path("src/onnx-runtime"),
    help="Worker directory that receives the generated runtime.",
  )
  parser.add_argument(
    "--no-install-worker",
    action="store_true",
    help="Build the runtime without copying it into the Worker source tree.",
  )
  parser.add_argument("--build-dir", type=Path)
  parser.add_argument("--ninja-executable", type=Path, required=True)
  parser.add_argument("--patch-executable", type=Path, required=True)
  parser.add_argument("--npm", default="npm")
  parser.add_argument("--jobs", type=int, default=max(1, min(16, os.cpu_count() or 1)))
  parser.add_argument("--skip-js-install", action="store_true")
  args = parser.parse_args(argv)
  if args.jobs <= 0:
    parser.error("--jobs must be positive")
  return args


def resolve_path(path: Path, workspace: Path) -> Path:
  path = path.expanduser()
  if not path.is_absolute():
    path = workspace / path
  return path.resolve()


def resolve_input(path: Path, workspace: Path) -> Path:
  path = path.expanduser()
  candidates = [path, workspace / path]
  runfiles_dir = os.environ.get("RUNFILES_DIR")
  if runfiles_dir:
    candidates.append(Path(runfiles_dir) / path)
  for candidate in candidates:
    if candidate.exists():
      return candidate.resolve()
  return resolve_path(path, workspace)


def prepare_source_copy(declared: Path, destination: Path, *, source_id: str) -> Path:
  if not declared.is_dir():
    raise SystemExit(f"Bazel-declared ONNX Runtime source does not exist: {declared}")
  stamp = destination / ".manafold-source-id"
  if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == source_id:
    return destination
  if destination.exists():
    shutil.rmtree(destination)
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copytree(declared, destination, symlinks=False)
  stamp.write_text(source_id + "\n", encoding="utf-8")
  return destination


def read_ort_version(source: Path) -> str:
  version_file = source / "VERSION_NUMBER"
  if not version_file.is_file():
    raise SystemExit(f"Not an ONNX Runtime source tree: {source}")
  version = version_file.read_text(encoding="utf-8").strip()
  missing = [source / "cmake" / "external" / "emsdk" / "emsdk"]
  missing = [path for path in missing if not path.is_file()]
  if missing:
    raise SystemExit(
      "The Bazel-declared ONNX Runtime source is missing recursive submodules."
    )
  return version


def resolve_executable(value: str, *, option: str) -> Path:
  candidate = Path(value).expanduser()
  resolved = str(candidate.resolve()) if candidate.is_file() else shutil.which(value)
  if resolved is None:
    raise SystemExit(
      f"Required executable `{value}` was not found. Install it or pass "
      f"{option}."
    )
  return Path(resolved).resolve()


def build_runtime(
  *,
  source: Path,
  build_root: Path,
  ops_config: Path,
  build_python: Path,
  ninja: Path,
  patch: Path,
  jobs: int,
) -> None:
  if not ops_config.is_file():
    raise SystemExit(f"Operator configuration does not exist: {ops_config}")
  command = [
    str(build_python), str(source / "tools" / "ci_build" / "build.py"),
    "--build_dir", str(build_root),
    "--config", "MinSizeRel",
    "--build_wasm",
    "--enable_wasm_simd",
    "--skip_tests",
    "--skip_pip_install",
    "--skip_submodule_sync",
    "--enable_wasm_api_exception_catching",
    "--disable_rtti",
    "--disable_ml_ops",
    "--disable_contrib_ops",
    "--disable_generation_ops",
    "--include_ops_by_config", str(ops_config),
    "--cmake_generator", "Ninja",
    "--cmake_extra_defines",
    f"CMAKE_MAKE_PROGRAM={ninja}",
    f"Patch_EXECUTABLE={patch}",
    "CMAKE_CXX_FLAGS=-Wno-pass-failed",
    "--parallel", str(jobs),
  ]
  environment = os.environ.copy()
  environment["PYTHONPATH"] = os.pathsep.join(sys.path)
  run(command, cwd=source, env=environment)


def package_web_runtime(
  *,
  source: Path,
  build_root: Path,
  output_dir: Path,
  version: str,
  npm: Path,
  install_js_dependencies: bool,
) -> dict[str, Path]:
  build_output = build_root / "MinSizeRel"
  wasm = build_output / "ort-wasm-simd.wasm"
  loader = build_output / "ort-wasm-simd.mjs"
  if not wasm.is_file() or not loader.is_file():
    raise SystemExit(f"Reduced runtime outputs were not produced in {build_output}")

  web_dir = source / "js" / "web"
  dist_dir = web_dir / "dist"
  dist_dir.mkdir(parents=True, exist_ok=True)
  adapted_loader = adapt_loader(loader.read_text(encoding="utf-8"))
  loader_names = (
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.jspi.mjs",
  )
  for name in loader_names:
    (dist_dir / name).write_text(adapted_loader, encoding="utf-8")
  shutil.copy2(wasm, dist_dir / "ort-wasm-simd-threaded.wasm")

  if install_js_dependencies:
    for directory in (source / "js", source / "js" / "common", web_dir):
      run([str(npm), "ci", "--no-audit", "--no-fund"], cwd=directory)
  run([str(npm), "run", "build", "--", "--bundle-mode=prod"], cwd=web_dir)

  output_dir.mkdir(parents=True, exist_ok=True)
  bundle_output = output_dir / "ort.wasm.cloudflare.mjs"
  wasm_output = output_dir / "ort-wasm-simd.wasm"
  shutil.copy2(dist_dir / "ort.wasm.bundle.min.mjs", bundle_output)
  shutil.copy2(wasm, wasm_output)
  manifest = output_dir / "runtime_manifest.json"
  manifest.write_text(json.dumps({
    "onnxruntime_version": version,
    "build": "operator-reduced-wasm-simd-min-size-rel",
    "cloudflare_precompiled_wasm": True,
    "files": {
      bundle_output.name: file_metadata(bundle_output),
      wasm_output.name: file_metadata(wasm_output),
    },
  }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return {"javascript": bundle_output, "wasm": wasm_output, "manifest": manifest}


def adapt_loader(source: str) -> str:
  binary_replacements = (
    ("if(a==N&&x)a=new Uint8Array(x);", "x"),
    ("if(a==N&&B)a=new Uint8Array(B);", "B"),
  )
  matched = [
    (old, variable)
    for old, variable in binary_replacements
    if source.count(old) == 1
  ]
  if len(matched) != 1:
    raise SystemExit(
      "The pinned ONNX Runtime loader no longer matches the Cloudflare "
      "wasmBinary adapter."
    )
  old, variable = matched[0]
  source = source.replace(
    old,
    f"if(a==N&&{variable})a={variable} instanceof WebAssembly.Module?"
    f"{variable}:new Uint8Array({variable});",
  )

  instantiate = "return await WebAssembly.instantiate(c,b)"
  if source.count(instantiate) != 1:
    raise SystemExit(
      "The pinned ONNX Runtime loader no longer matches the Cloudflare "
      "WebAssembly.instantiate adapter."
    )
  source = source.replace(
    instantiate,
    "let cf=await WebAssembly.instantiate(c,b);"
    "return c instanceof WebAssembly.Module?{instance:cf,module:c}:cf",
  )
  # ORT's packager expects one worker construction site. This dead expression
  # is removed by Terser and lets a single-threaded loader use that packager.
  return "if(false)new Worker(new URL(import.meta.url),{});\n" + source


def install_worker_runtime(artifacts: dict[str, Path], worker_dir: Path) -> None:
  vendor = worker_dir / "src" / "vendor"
  model = worker_dir / "src" / "model"
  vendor.mkdir(parents=True, exist_ok=True)
  model.mkdir(parents=True, exist_ok=True)
  shutil.copy2(artifacts["javascript"], vendor / "ort.wasm.cloudflare.mjs")
  shutil.copy2(artifacts["wasm"], model / "ort-wasm-simd-threaded.wasm")
  shutil.copy2(artifacts["manifest"], model / "runtime_manifest.json")


def file_metadata(path: Path) -> dict[str, object]:
  data = path.read_bytes()
  return {
    "path": str(path),
    "bytes": len(data),
    "gzip_bytes": len(gzip.compress(data, compresslevel=9)),
    "sha256": hashlib.sha256(data).hexdigest(),
  }


def run(
  command: list[str],
  *,
  cwd: Path | None = None,
  env: dict[str, str] | None = None,
) -> None:
  print("+ " + " ".join(command), flush=True)
  subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
  raise SystemExit(main())
