"""This script replaces all-atom coordinates with C_a-only in each mdCATH domain.

Usage:
    python scripts/prepare_mdcath.py /path/to/mdCATH/data
"""

import h5py
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from platito.data.constants import MDCATH_ALL_TEMPS, MDCATH_ALL_REPLICAS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to mdCATH/data/ directory")
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).glob("mdcath_dataset_*.h5"))
    if not files:
        raise SystemExit(f"No HDF5 files found in {args.data_dir}")

    errors = []
    for h5_path in tqdm(files):
        domain = h5_path.stem.replace("mdcath_dataset_", "")
        try:
            with h5py.File(h5_path, "r+") as f:
                pdb_lines = f[domain]["pdbProteinAtoms"][()].decode("utf-8").split("\n")[1:-3]
                atomtypes = [line.split()[2] for line in pdb_lines]
                ca_idx = np.where(np.array(atomtypes) == "CA")[0]
                for temp in MDCATH_ALL_TEMPS:
                    for repl in MDCATH_ALL_REPLICAS:
                        grp = f[domain][temp][repl]
                        if "ca_coords" in grp:
                            del grp["ca_coords"]
                        coords = grp["coords"][:]
                        grp.create_dataset(
                            "ca_coords",
                            data=coords[:, ca_idx, :],
                            compression="gzip",
                            compression_opts=4,
                        )
                        del grp["coords"]
        except Exception as e:
            errors.append(f"{domain}: {e}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(" ", e)
    else:
        print("Done.")


if __name__ == "__main__":
    main()
