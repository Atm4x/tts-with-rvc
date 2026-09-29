from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from urllib.error import HTTPError
from urllib.request import urlopen
import venv
import zipfile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
PROJECTS = {
    "fish-speech-lib": ("fish_speech_lib", "releases", "pypi-fish", "Atm4x/Fish-speech-pipeline"),
    "tts-with-rvc": ("tts_with_rvc", "releases", "pypi-rvc", "Atm4x/tts-with-rvc"),
    "tts-with-rvc-onnx": ("tts_with_rvc", "releases-onnx", "pypi-onnx", "Atm4x/tts-with-rvc"),
}


def project():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)["project"]
    name = canonicalize_name(data["name"])
    if name not in PROJECTS:
        raise ValueError(f"Unsupported project: {name}")
    version = Version(data["version"])
    if str(version) != data["version"]:
        raise ValueError("Use a normalized PEP 440 version")
    return name, version, PROJECTS[name]


def runtime_version(package):
    tree = ast.parse((ROOT / package / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Package must define a literal __version__")


def published_versions(name):
    try:
        with urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as response:
            data = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return []
        raise
    return [Version(value) for value in data["releases"]]


def check(release=False):
    name, version, (package, branch, environment, repository) = project()
    if runtime_version(package) != str(version):
        raise ValueError("pyproject.toml and package __version__ disagree; run the version command")
    if release:
        actual_branch = os.environ.get("GITHUB_REF_NAME") or subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip()
        if actual_branch != branch:
            raise ValueError(f"{name} can only be published from {branch}, got {actual_branch}")
        if os.environ.get("GITHUB_REPOSITORY", repository).lower() != repository.lower():
            raise ValueError(f"Unexpected publishing repository; expected {repository}")
        versions = published_versions(name)
        if version in versions:
            raise ValueError(f"{name} {version} already exists on PyPI; bump the version")
        stable = [value for value in versions if not value.is_prerelease and not value.is_devrelease]
        if stable and version <= max(stable):
            raise ValueError(f"Version must be greater than the latest stable release {max(stable)}")
    outputs = {
        "name": name, "version": str(version), "package": package,
        "environment": environment, "tag": f"{name}/v{version}",
    }
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            for key, value in outputs.items():
                stream.write(f"{key}={value}\n")
    print(json.dumps(outputs, indent=2))


def set_version(value=None, bump=False):
    _, old, (package, _, _, _) = project()
    if bump:
        if old.is_prerelease or old.is_devrelease or old.is_postrelease:
            raise ValueError("Use --set for prerelease, dev or post versions")
        parts = list(old.release)
        parts[-1] += 1
        value = ".".join(map(str, parts))
    new = Version(value)
    if str(new) != value or new <= old:
        raise ValueError(f"Use a normalized version greater than {old}")
    paths = (ROOT / "pyproject.toml", ROOT / package / "__init__.py")
    updates = []
    for path, pattern, replacement in (
        (paths[0], r'^version[ \t]*=[ \t]*"[^"\n]+"[ \t]*$', f'version = "{new}"'),
        (paths[1], r'^__version__\s*=\s*[\'"][^\'"\n]+[\'"]\s*$', f'__version__ = "{new}"'),
    ):
        content, count = re.subn(pattern, replacement, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected exactly one version assignment in {path}")
        updates.append((path, content))
    for path, content in updates:
        path.write_text(content, encoding="utf-8", newline="\n")
    check()


def verify(dist):
    name, version, (package, _, _, _) = project()
    wheels = list(dist.glob("*.whl"))
    sources = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1 or len(list(dist.iterdir())) != 2:
        raise ValueError("Expected exactly one wheel and one sdist; use a clean output directory")
    expected = {path.relative_to(ROOT).as_posix() for path in (ROOT / package).rglob("*.py")}
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        metadata_paths = [value for value in names if value.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Expected exactly one wheel METADATA")
        wheel_metadata = archive.read(metadata_paths[0])
        missing = expected - names
        if missing:
            raise ValueError(f"Wheel is missing Python modules: {sorted(missing)}")
        for relative in expected:
            if archive.read(relative) != (ROOT / relative).read_bytes():
                raise ValueError(f"Wheel contains stale source: {relative}")
        unexpected = {value for value in names if value.endswith(".py")} - expected
        if unexpected:
            raise ValueError(f"Wheel contains unexpected Python modules: {sorted(unexpected)}")
        if not any(value.endswith("/licenses/LICENSE") for value in names):
            raise ValueError("Wheel is missing LICENSE")
        if (ROOT / "NOTICE").exists() and not any(value.endswith("/licenses/NOTICE") for value in names):
            raise ValueError("Wheel is missing NOTICE")
    with tarfile.open(sources[0], "r:gz") as archive:
        names = {"/".join(value.split("/")[1:]) for value in archive.getnames()}
        if not expected <= names or not {"pyproject.toml", "README.md", "LICENSE"} <= names:
            raise ValueError("sdist is missing source files or packaging metadata")
        member = next(value for value in archive.getmembers() if value.name.count("/") == 1 and value.name.endswith("/PKG-INFO"))
        source_metadata = archive.extractfile(member).read()
    with (ROOT / "pyproject.toml").open("rb") as stream:
        expected_python = tomllib.load(stream)["project"]["requires-python"]
    for raw in (wheel_metadata, source_metadata):
        metadata = BytesParser().parsebytes(raw)
        if canonicalize_name(metadata["Name"]) != name or Version(metadata["Version"]) != version:
            raise ValueError("Distribution name/version disagrees with pyproject.toml")
        if SpecifierSet(metadata["Requires-Python"]) != SpecifierSet(expected_python):
            raise ValueError("Distribution Python requirement disagrees with pyproject.toml")
    print(f"Verified {name} {version}: {len(expected)} Python modules, wheel + sdist + license files")


def smoke(dist):
    name, version, (package, _, _, _) = project()
    wheel = next(dist.glob("*.whl")).resolve()
    with tempfile.TemporaryDirectory(prefix="package-smoke-") as directory:
        env = Path(directory) / "venv"
        venv.create(env, with_pip=True)
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--disable-pip-version-check", str(wheel)], check=True)
        code = (
            "import importlib.metadata as m, importlib.util as u, pathlib, sys; "
            f"assert m.version({name!r}) == {str(version)!r}; "
            f"p = pathlib.Path(u.find_spec({package!r}).origin).resolve(); "
            "assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()), p; "
            "print('Installed wheel metadata and package location verified:', p)"
        )
        subprocess.run([str(python), "-I", "-c", code], check=True, cwd=directory)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("check")
    command.add_argument("--release", action="store_true")
    command = commands.add_parser("version")
    versions = command.add_mutually_exclusive_group(required=True)
    versions.add_argument("--set")
    versions.add_argument("--bump", action="store_true")
    for action in ("verify", "smoke"):
        commands.add_parser(action).add_argument("--dist", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    if args.command == "check":
        check(args.release)
    elif args.command == "version":
        set_version(args.set, args.bump)
    elif args.command == "verify":
        verify(args.dist)
    else:
        smoke(args.dist)


if __name__ == "__main__":
    main()
