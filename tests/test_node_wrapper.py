"""Tests for the npx / Node.js llmtap CLI wrapper."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER_SCRIPT = REPO_ROOT / "bin" / "llmtap.js"


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