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
    _write_executable(
        bin_dir / "gcloud",
        """#!/bin/sh
printf '%s\n' "$@" > "$TMPDIR/captured-gcloud-args"
for arg in "$@"; do
  case "$arg" in
    --metadata-from-file=startup-script=*)
      cp "${arg#--metadata-from-file=startup-script=}" "$TMPDIR/captured-startup"
      ;;
  esac
done
exit 0
""",
    )

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
            "--resume",
            "--boot-disk-size-gb",
            "160",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "bt-startup.ABC123").exists()
    startup = (tmp_path / "captured-startup").read_text(encoding="utf-8")
    assert "for ARTIFACT in latest.pt best.pt metrics.jsonl metrics_latest.json; do" in startup
    assert 'gsutil -q cp "$RUN/$ARTIFACT" "/opt/bt/runs/test-run/$ARTIFACT"' in startup
    assert "for ARTIFACT in metrics_best_full.json metrics_test.json; do" in startup
    assert "pymedphys numba" in startup
    assert 'gsutil -q cp "/opt/bt/runs/test-run/$ARTIFACT" "$RUN/$ARTIFACT"' in startup
    gcloud_args = (tmp_path / "captured-gcloud-args").read_text(encoding="utf-8").splitlines()
    assert "--boot-disk-size=160GB" in gcloud_args


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
