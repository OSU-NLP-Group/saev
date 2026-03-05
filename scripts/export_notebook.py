"""Export examples/inference.py (marimo) to examples/inference.ipynb (Jupyter).

Adds a pip install cell pinned to the current git commit and a Colab badge.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARIMO_SRC = ROOT / "examples" / "inference.py"
IPYNB_DST = ROOT / "examples" / "inference.ipynb"
REPO = "OSU-NLP-Group/saev"


def get_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, f"git rev-parse failed: {result.stderr}"
    return result.stdout.strip()


def export_marimo() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "marimo",
            "export",
            "ipynb",
            str(MARIMO_SRC),
            "-o",
            str(IPYNB_DST),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source],
    }


def postprocess(commit: str) -> None:
    with open(IPYNB_DST) as fd:
        nb = json.load(fd)

    pip_cell = make_code_cell(
        f'!pip install "saev @ git+https://github.com/{REPO}@{commit}"'
    )

    badge_url = f"https://colab.research.google.com/github/{REPO}/blob/main/examples/inference.ipynb"
    badge_md = (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({badge_url})\n\n"
        "<!-- DO NOT EDIT -- generated from examples/inference.py -->"
    )
    badge_cell = make_markdown_cell(badge_md)

    nb["cells"] = [badge_cell, pip_cell] + nb["cells"]

    with open(IPYNB_DST, "w") as fd:
        json.dump(nb, fd, indent=1)
        fd.write("\n")

    print(f"Wrote {IPYNB_DST} (pinned to {commit[:8]})")


def main():
    print("Exporting marimo notebook to ipynb...")
    export_marimo()
    commit = get_commit_hash()
    postprocess(commit)


if __name__ == "__main__":
    main()
