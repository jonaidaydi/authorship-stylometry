"""Execute all three notebooks in fresh kernels and keep originals unchanged.

Run after prepare_data.py --federalist. Executed copies and the temporary
kernel definition stay in the ignored verification_outputs directory.
"""

import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "verification_outputs"
    kernels = output / "kernels"
    spec = kernels / "stylometry-check"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Stylometry verification", "language": "python",
        "env": {"MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8",
                "IPYTHONDIR": str(output / "ipython"),
                "JUPYTER_RUNTIME_DIR": str(output / "runtime")},
    }), encoding="utf-8")
    manager = KernelSpecManager(kernel_dirs=[str(kernels)])
    for filename in ["reuter HKA, HCA.ipynb", "HKA Federalist Papiere.ipynb", "HCA Federalist Papiere.ipynb"]:
        notebook = nbformat.read(ROOT / filename, as_version=4)
        nbformat.validate(notebook)
        kernel = KernelManager(kernel_name="stylometry-check", kernel_spec_manager=manager)
        client = NotebookClient(notebook, km=kernel, timeout=600, resources={"metadata": {"path": str(ROOT)}})
        print(f"Running {filename}...", flush=True)
        try:
            client.execute(cwd=str(ROOT))
        finally:
            if kernel.has_kernel:
                kernel.shutdown_kernel(now=True)
        nbformat.write(notebook, output / filename)
        print(f"PASS: {filename}", flush=True)


if __name__ == "__main__":
    main()
