#!/usr/bin/env python3
"""Sanitize and push an Enveda CASMI26 notebook to Kaggle via kagglesdk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest, KernelExecutionType

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "kernels.json"


def sanitize_notebook(path: Path) -> str:
    nb = json.loads(path.read_text())
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 4
    for cell in nb.get("cells", []):
        cell.pop("id", None)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return json.dumps(nb, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", choices=["cpu-baseline-v17", "kozykappa-e180x-npx"])
    ap.add_argument("--run", action="store_true", help="SAVE_AND_RUN_ALL instead of QUICK_SAVE")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())["kernels"][args.kernel]
    comp = json.loads(CONFIG.read_text())["competition"]
    text = sanitize_notebook(ROOT / cfg["notebook"])

    req = ApiSaveKernelRequest()
    req.slug = f"kulrajsabharwal/{cfg['slug']}"
    req.new_title = cfg["title"]
    req.text = text
    req.language = "python"
    req.kernel_type = "notebook"
    req.is_private = cfg.get("private", True)
    req.enable_gpu = cfg.get("gpu", False)
    if cfg.get("machine_shape"):
        req.machine_shape = cfg["machine_shape"]
    req.enable_internet = False
    req.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL if args.run else KernelExecutionType.QUICK_SAVE
    req.session_timeout_seconds = cfg.get("timeout_seconds", 32400)
    req.competition_data_sources = [comp]
    req.dataset_data_sources = cfg.get("datasets", [])
    req.kernel_data_sources = cfg.get("kernel_sources", [])
    req.category_ids = cfg.get("categories", [])

    resp = KaggleClient().kernels.kernels_api_client.save_kernel(req)
    print(json.dumps(resp.to_dict(), indent=2))


if __name__ == "__main__":
    main()
