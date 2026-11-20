"""Tests for the npx / Node.js llmtap CLI wrapper."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER_SCRIPT = REPO_ROOT / "bin" / "llmtap.js"


def test_package_json_matches_the_python_package() -> None:
    """npm metadata must not drift from PyPI: same version, and a name this project owns.

    `llmtap` on npm (like on PyPI) belongs to someone else, so `npx llmtap` would run their code."""
    import json
    import re

    pkg = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    py_version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    from llmtap import __version__

    assert pkg["name"] == "llmtap-proxy"
    assert pkg["version"] == py_version == __version__
    assert pkg["bin"] == {"llmtap": "./bin/llmtap.js"}
    assert pkg["license"] == "MIT"


@pytest.fixture(scope="module")
def node_executable() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed on this system")
    return node


def test_node_wrapper_version(node_executable: str) -> None:
    """node bin/llmtap.js version should exit 0 and print the version."""
    proc = subprocess.run(
        [node_executable, str(WRAPPER_SCRIPT), "version"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "0." in proc.stdout


def test_node_wrapper_argument_forwarding(node_executable: str) -> None:
    """node bin/llmtap.js prices claude should forward arguments correctly."""
    proc = subprocess.run(
        [node_executable, str(WRAPPER_SCRIPT), "prices", "claude"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert proc.stdout is not None
    assert "claude" in proc.stdout.lower()


def test_node_wrapper_help(node_executable: str) -> None:
    """node bin/llmtap.js --help should output llmtap usage help."""
    proc = subprocess.run(
        [node_executable, str(WRAPPER_SCRIPT), "--help"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "llmtap" in proc.stdout