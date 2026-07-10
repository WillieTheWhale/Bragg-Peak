from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_launcher_uses_portable_cleaned_up_startup_script(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "mktemp",
        """#!/bin/sh
case "$1" in
  *XXXXXX) ;;
  *) echo "template must end in XXXXXX" >&2; exit 1 ;;
esac
path="${1%XXXXXX}ABC123"
: > "$path"
printf '%s\n' "$path"
""",
    )
    _write_executable(bin_dir / "gcloud", "#!/bin/sh\nexit 0\n")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["TMPDIR"] = str(tmp_path)
    result = subprocess.run(
        [
            str(repo_root / "scripts" / "gcp_iter_launch.sh"),
            "--run-name",
            "test-run",
            "--train-args",
            "--epochs 1",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "bt-startup.ABC123").exists()


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
