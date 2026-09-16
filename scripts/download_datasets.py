"""Manual downloader for the public field datasets (Stage II).

These datasets are large and hosted externally, so they are **not** fetched
during environment setup or CI.  Run this script by hand when you want to work
with real data:

    python scripts/download_datasets.py --dataset marmousi2 --dest data/marmousi2

Notes
-----
* AGL Elastic Marmousi (Vp, Vs, density, seismic):
  https://wiki.seg.org/wiki/AGL_Elastic_Marmousi
* Penobscot 3D (seismic, wells B-41 & L-30, horizons):
  https://wiki.seg.org/wiki/Penobscot_3D
  https://zenodo.org/records/1325077

Because the canonical download URLs change and some require accepting a licence
or using dGB/OpendTect, this script prints the authoritative source pages rather
than hard-coding fragile links.  Provide ``--url`` to fetch a specific archive.
"""

from __future__ import annotations

import argparse
import os
import urllib.request

SOURCES = {
    "marmousi2": "https://wiki.seg.org/wiki/AGL_Elastic_Marmousi",
    "penobscot": "https://wiki.seg.org/wiki/Penobscot_3D",
    "penobscot_zenodo": "https://zenodo.org/records/1325077",
    "mwd_spatial": "https://zenodo.org/records/10358374",
    "mwd_raw": "https://catalog.data.gov/dataset/data-from-st-project-21049-improving-subsurface-characterization-with-monitoring-while-dri",
    "mwd_ucs": "https://www.nature.com/articles/s41598-025-93111-4",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--dest", default="data")
    parser.add_argument("--url", help="direct archive URL to download")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    print(f"Dataset       : {args.dataset}")
    print(f"Source page   : {SOURCES[args.dataset]}")
    print(f"Destination   : {args.dest}")

    if not args.url:
        print(
            "\nNo --url provided. Open the source page above, obtain the direct\n"
            "download link (some require accepting a licence), then re-run with\n"
            "--url <link>."
        )
        return

    fname = os.path.join(args.dest, os.path.basename(args.url) or "download.bin")
    print(f"Downloading   : {args.url}\n           -> : {fname}")
    urllib.request.urlretrieve(args.url, fname)  # noqa: S310
    print("Done.")


if __name__ == "__main__":
    main()
