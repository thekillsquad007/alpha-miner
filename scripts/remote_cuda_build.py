#!/usr/bin/env python3
"""Build alpha-miner CUDA binary on a remote NVIDIA host via paramiko."""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko

HOST = os.environ.get("CUDA_SSH_HOST", "n1.de.clorecloud.net")
PORT = int(os.environ.get("CUDA_SSH_PORT", "1707"))
USER = os.environ.get("CUDA_SSH_USER", "root")
PASSWORD = os.environ.get("CUDA_SSH_PASSWORD", "pass")
REMOTE_DIR = os.environ.get("CUDA_REMOTE_DIR", "/tmp/alpha-miner-build")
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "dist" / "release"
BINARY_NAME = "alpha-miner-linux-cuda-x64"


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[ssh] connecting {USER}@{HOST}:{PORT} ...")
    client.connect(
        hostname=HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        timeout=60,
        allow_agent=False,
        look_for_keys=False,
        banner_timeout=60,
    )
    print("[ssh] connected")
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 3600) -> tuple[int, str, str]:
    print(f"[remote] $ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        # print last ~80 lines to keep log readable
        lines = out.strip().splitlines()
        for line in lines[-80:]:
            print(f"  | {line}")
    if err.strip() and code != 0:
        for line in err.strip().splitlines()[-40:]:
            print(f"  ! {line}", file=sys.stderr)
    return code, out, err


def make_source_tarball() -> Path:
    exclude_dirs = {
        ".git",
        "build",
        "build-mod",
        "build-hip",
        "build-hip-mod",
        "build-hip-rel",
        "build-linux",
        "build-linux-rel",
        "build-cuda",
        "build-win",
        "dist",
        "cmake",
    }
    tmp = Path(tempfile.mkdtemp(prefix="alpha-src-"))
    tar_path = tmp / "alpha-miner-src.tar.gz"
    print(f"[pack] creating {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith("build")]
            rel_root = Path(root).relative_to(REPO)
            # skip nested build dirs
            if any(p.startswith("build") for p in rel_root.parts):
                continue
            for f in files:
                full = Path(root) / f
                arc = Path("alpha-miner") / rel_root / f
                tar.add(full, arcname=str(arc))
    print(f"[pack] size={tar_path.stat().st_size / 1e6:.1f} MB")
    return tar_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = make_source_tarball()
    client = connect()
    try:
        sftp = client.open_sftp()
        run(client, f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}")
        remote_tar = f"{REMOTE_DIR}/src.tar.gz"
        print(f"[sftp] upload {tar_path} -> {remote_tar}")
        sftp.put(str(tar_path), remote_tar)

        # Probe toolchain
        code, out, _ = run(
            client,
            "which nvcc; nvcc --version 2>/dev/null | tail -1; "
            "nvidia-smi -L 2>/dev/null | head -5; "
            "which cmake g++ ; cmake --version | head -1; "
            "nproc; uname -a",
        )
        if "nvcc" not in out and code != 0:
            print("[error] remote probe failed", file=sys.stderr)

        build_script = r"""
set -euo pipefail
cd {remote}
rm -rf alpha-miner
tar xzf src.tar.gz
cd alpha-miner
export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.6/bin:/usr/local/cuda-12.4/bin:/usr/local/cuda-12.2/bin:/usr/local/cuda-12/bin:${{PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${{LD_LIBRARY_PATH:-}}"
# locate nvcc
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda* /opt/cuda*; do
    if [ -x "$d/bin/nvcc" ]; then export PATH="$d/bin:$PATH"; break; fi
  done
fi
command -v nvcc
nvcc --version | tail -1
# deps
if ! command -v cmake >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake 2>&1 | tail -5
fi
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DALPHA_MINER_CUDA=ON -DALPHA_MINER_HIP=OFF -DALPHA_MINER_OPENCL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100;120"
cmake --build build-cuda -j"$(nproc)"
./build-cuda/alpha-miner --help | head -5
./build-cuda/alpha-miner --benchmark -c lattica || true
./build-cuda/alpha-miner -l || true
strip build-cuda/alpha-miner || true
ls -la build-cuda/alpha-miner
""".format(
            remote=REMOTE_DIR
        )
        # write build script remotely
        remote_sh = f"{REMOTE_DIR}/build.sh"
        with sftp.file(remote_sh, "w") as f:
            f.write(build_script)
        run(client, f"chmod +x {remote_sh}")
        code, out, err = run(client, f"bash {remote_sh}", timeout=3600)
        if code != 0:
            print(f"[error] remote build failed exit={code}", file=sys.stderr)
            return code or 1

        remote_bin = f"{REMOTE_DIR}/alpha-miner/build-cuda/alpha-miner"
        local_bin = OUT_DIR / BINARY_NAME
        print(f"[sftp] download {remote_bin} -> {local_bin}")
        sftp.get(remote_bin, str(local_bin))
        os.chmod(local_bin, 0o755)
        print(f"[ok] {local_bin} ({local_bin.stat().st_size} bytes)")
        sftp.close()
        return 0
    finally:
        client.close()
        try:
            tar_path.unlink(missing_ok=True)
            tar_path.parent.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
