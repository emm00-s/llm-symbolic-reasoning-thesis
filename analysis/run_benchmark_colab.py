"""Reference workflow for reproducing the behavioral benchmark on Google Colab.

This module documents (and provides a thin programmatic wrapper around) the
end-to-end pipeline that produced the Qwen-2.5-3B and Llama-3.2-3B result CSVs
in ``results/``. It is intended as an executable record of the Colab workflow,
not as a substitute for ``src.run`` / ``src.analyze``.

Notes
-----
- The full Qwen and Llama runs require a GPU. The smoke tests can be executed
  on CPU but are still slow.
- ``meta-llama/Llama-3.2-3B-Instruct`` is a gated model and requires a valid
  Hugging Face token (via ``huggingface-cli login`` or the ``HF_TOKEN``
  environment variable).
- All shell commands assume the working directory is the repository root.

Typical sequence
----------------
1. Install dependencies:        ``pip install -r requirements.txt``
2. Validate dataset and Z3:     ``python -m src.dataset`` and ``python -m src.solvers``
3. Smoke test:                  ``python -m src.run --seeds 2 --tag smoke``
4. Qwen full run:               ``python -m src.run --model-name Qwen/Qwen2.5-3B-Instruct --seeds 10 --tag qwen3b``
5. Authenticate with HF for Llama (gated).
6. Llama full run:              ``python -m src.run --model-name meta-llama/Llama-3.2-3B-Instruct --seeds 10 --tag llama32_3b``
7. Generate per-model summaries: ``python -m src.analyze results/<RESULT_FILE>.csv``
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def run(cmd: list[str]) -> None:
    """Run a subprocess and stream its output."""
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def install_requirements() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def validate_pipeline() -> None:
    run([sys.executable, "-m", "py_compile", *map(str, (REPO_ROOT / "src").glob("*.py"))])
    run([sys.executable, "-m", "src.dataset"])
    run([sys.executable, "-m", "src.solvers"])


def smoke_test(tag: str = "smoke") -> None:
    run([sys.executable, "-m", "src.run", "--seeds", "2",
         "--temperature", "0.7", "--top-p", "1.0", "--tag", tag])


def run_qwen(seeds: int = 10, tag: str = "qwen3b") -> None:
    run([sys.executable, "-m", "src.run",
         "--model-name", "Qwen/Qwen2.5-3B-Instruct",
         "--seeds", str(seeds),
         "--temperature", "0.7", "--top-p", "1.0",
         "--tag", tag])


def authenticate_huggingface() -> None:
    """Interactive HF login. Required for the gated Llama-3.2 model."""
    from getpass import getpass
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN") or getpass("Paste your Hugging Face token: ").strip()
    login(token=token, add_to_git_credential=True)
    print(HfApi().whoami())


def run_llama(seeds: int = 10, tag: str = "llama32_3b") -> None:
    run([sys.executable, "-m", "src.run",
         "--model-name", "meta-llama/Llama-3.2-3B-Instruct",
         "--seeds", str(seeds),
         "--temperature", "0.7", "--top-p", "1.0",
         "--tag", tag])


def latest_csv(tag: str) -> Path:
    pattern = str(RESULTS_DIR / f"results_{tag}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No results CSV matching {pattern}")
    return Path(max(matches, key=os.path.getmtime))


def analyze_to_txt(csv_path: Path, out_path: Path) -> None:
    print(f"Analyzing {csv_path} -> {out_path}", flush=True)
    with out_path.open("w") as fh:
        subprocess.run(
            [sys.executable, "-m", "src.analyze", str(csv_path)],
            stdout=fh, stderr=subprocess.STDOUT, check=True, cwd=REPO_ROOT,
        )


def analyze_qwen_and_llama() -> None:
    qwen_csv = latest_csv("qwen3b")
    llama_csv = latest_csv("llama32_3b")
    analyze_to_txt(qwen_csv, RESULTS_DIR / "analysis_qwen3b.txt")
    analyze_to_txt(llama_csv, RESULTS_DIR / "analysis_llama32_3b.txt")


def main() -> None:
    install_requirements()
    validate_pipeline()
    smoke_test()
    run_qwen()
    authenticate_huggingface()
    run_llama()
    analyze_qwen_and_llama()


if __name__ == "__main__":
    main()
