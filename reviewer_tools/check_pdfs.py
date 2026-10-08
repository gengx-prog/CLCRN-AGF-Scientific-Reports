"""Open and render every page of the supplied manuscript PDFs without repairs."""
import argparse
import json
from pathlib import Path
import sys

import pymupdf

from reviewer_verify import require, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "pymupdf": pymupdf.VersionBind, "files": []}
    status = 0
    try:
        for relative in ("Scientific_Reports_submission/main.pdf", "Supplementary_Information/supplement.pdf"):
            path = args.repo_root / relative
            require(path.is_file(), f"Missing manuscript PDF: {relative}")
            with path.open("rb") as stream:
                require(stream.read(5) == b"%PDF-", f"Invalid PDF header: {relative}")
            with pymupdf.open(path) as document:
                require(not document.is_repaired, f"PDF needs repair: {relative}")
                require(not document.needs_pass and len(document) > 0, f"Encrypted/empty PDF: {relative}")
                for index, page in enumerate(document, 1):
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5), alpha=False)
                    require(pixmap.width > 0 and pixmap.height > 0 and bool(pixmap.samples),
                            f"Page cannot render: {relative}, page {index}")
                report["files"].append({"file": relative, "sha256": sha256(path), "pages": len(document),
                                        "all_pages_rendered": True, "repair_required": False})
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        status = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
