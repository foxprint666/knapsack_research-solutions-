"""
azure_v1_to_matrix.py
Converts Azure Public Dataset V1 (2017) CPU readings + vmtable into
the (N_workloads x T_bins) matrix needed by m0_gate_metric.py.

Schema (NO HEADER ROW in the files):
  vmtable columns (positional):
    0: vm_id (STRING_HASH)
    1: subscription_id (STRING_HASH)
    2: deployment_id (STRING_HASH)
    3: ts_created (INTEGER, seconds)
    4: ts_deleted (INTEGER, seconds, may be empty)
    5: max_cpu (DOUBLE)
    6: avg_cpu (DOUBLE)
    7: p95_max_cpu (DOUBLE)
    8: vm_category (STRING)   <- "Interactive", "Delay-insensitive", "Unknown", or empty
    9: vm_core_count (INTEGER or ">N" string)
   10: vm_memory_gb (DOUBLE)

  vm_cpu_readings columns (positional):
    0: timestamp (INTEGER, seconds from 0, every 300 s)
    1: vm_id (STRING_HASH)
    2: min_cpu (DOUBLE)
    3: max_cpu (DOUBLE)
    4: avg_cpu (DOUBLE)

EDIT these lists if your files have a different column order:
"""

# ----- CONFIGURABLE COLUMN INDICES -----
# vmtable: column indices (0-based) for the fields we need
VC_VMID  = 0
VC_TSCR  = 3    # timestamp vm created
VC_TSDEL = 4    # timestamp vm deleted (may be NaN)
VC_CAT   = 8    # vm_category string
VC_CORES = 9    # vm_virtual_core_count

# readings: column indices (0-based)
RC_TS    = 0    # timestamp seconds
RC_VMID  = 1    # vm id
RC_AVG   = 4    # avg cpu

# ----------------------------------------

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

BIN_SEC = 300       # 5-minute bins
BINS_PER_DAY = 86400 // BIN_SEC   # 288


def parse_cores(val):
    """Map Azure core count field to a numeric value.
    '>24' -> 32, empty/NaN -> 2."""
    if pd.isna(val):
        return 2.0
    s = str(val).strip().lstrip(">").lstrip("<")
    try:
        return float(s)
    except ValueError:
        return 2.0


def load_vmtable(path):
    """Load vmtable, return DataFrame with columns: vm_id, ts_created, ts_deleted, category, cores."""
    print(f"[vmtable] Loading {path} ...")
    df = pd.read_csv(
        path,
        header=None,
        usecols=[VC_VMID, VC_TSCR, VC_TSDEL, VC_CAT, VC_CORES],
        dtype={VC_VMID: str, VC_CAT: str},
    )
    df.columns = ["vm_id", "ts_created", "ts_deleted", "category", "cores_raw"]
    df["ts_created"] = pd.to_numeric(df["ts_created"], errors="coerce")
    df["ts_deleted"] = pd.to_numeric(df["ts_deleted"], errors="coerce")
    df["cores"] = df["cores_raw"].apply(parse_cores)
    df["category"] = df["category"].fillna("Unknown").str.strip()
    df = df.drop(columns="cores_raw").drop_duplicates(subset="vm_id")
    print(f"[vmtable] {len(df):,} unique VMs loaded.")
    return df.set_index("vm_id")


def inspect_readings(pattern):
    """Print summary stats on the readings files without building the matrix."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[inspect] No files matched: {pattern}")
        return
    print(f"[inspect] Found {len(files)} file(s): {[os.path.basename(f) for f in files]}")
    for fp in files:
        chunk = pd.read_csv(fp, header=None, usecols=[RC_TS, RC_VMID],
                            dtype={RC_VMID: str}, nrows=200_000)
        chunk.columns = ["ts", "vm_id"]
        ts_min, ts_max = chunk["ts"].min(), chunk["ts"].max()
        n_vms = chunk["vm_id"].nunique()
        print(f"  {os.path.basename(fp)}: ts [{ts_min:,} – {ts_max:,}] s  "
              f"({(ts_max-ts_min)/86400:.1f} days in first 200k rows)  "
              f"unique VMs in sample: {n_vms:,}")


def build_matrix(vmtable_df, reading_files, day_start, days, max_vms,
                 max_missing_frac=0.10):
    """
    Build aligned matrix for VMs that are alive across the full window
    [day_start*86400, (day_start+days)*86400).

    Returns: matrix (N, T), weights (N,), labels (N,), vm_ids (N,)
    """
    t0 = day_start * 86400
    t1 = t0 + days * 86400
    T = days * BINS_PER_DAY

    # -- Select VMs alive for the full window --
    vmt = vmtable_df.copy()
    alive = (vmt["ts_created"] <= t0) & (
        vmt["ts_deleted"].isna() | (vmt["ts_deleted"] >= t1)
    )
    vmt = vmt[alive]
    print(f"[build] Window days {day_start}–{day_start+days}: {len(vmt):,} VMs alive for full window.")

    if len(vmt) == 0:
        print("[build] ERROR: No VMs alive for the entire window. Try --day-start 0 or download more files.")
        sys.exit(1)

    # Subsample if needed (take a reproducible random sample for speed)
    if len(vmt) > max_vms:
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(vmt), size=max_vms, replace=False)
        vmt = vmt.iloc[chosen]
        print(f"[build] Subsampled to {max_vms:,} VMs.")

    vm_ids_list = list(vmt.index)
    vm_id_set = set(vm_ids_list)
    vm_idx = {v: i for i, v in enumerate(vm_ids_list)}
    N = len(vm_ids_list)
    min_readings = int(T * (1.0 - max_missing_frac))

    # -- Accumulate readings into sum/count arrays --
    sums = np.zeros((N, T), dtype=np.float64)
    counts = np.zeros((N, T), dtype=np.int32)

    for fp in reading_files:
        print(f"[build] Reading {os.path.basename(fp)} ...")
        for chunk in pd.read_csv(fp, header=None, usecols=[RC_TS, RC_VMID, RC_AVG],
                                  dtype={RC_VMID: str}, chunksize=500_000):
            chunk.columns = ["ts", "vm_id", "avg"]
            # filter to window and known VMs
            chunk = chunk[(chunk["ts"] >= t0) & (chunk["ts"] < t1)]
            chunk = chunk[chunk["vm_id"].isin(vm_id_set)]
            if chunk.empty:
                continue
            bin_idx = ((chunk["ts"] - t0) // BIN_SEC).astype(int).clip(0, T - 1)
            row_idx = chunk["vm_id"].map(vm_idx)
            valid = row_idx.notna()
            chunk = chunk[valid]; bin_idx = bin_idx[valid]; row_idx = row_idx[valid].astype(int)
            np.add.at(sums,   (row_idx.values, bin_idx.values), chunk["avg"].values)
            np.add.at(counts, (row_idx.values, bin_idx.values), 1)

    # -- Build matrix: mean where count>0, forward-fill gaps, drop sparse VMs --
    matrix = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan).astype(np.float32)

    # forward-fill then backward-fill NaN within each row
    df_mat = pd.DataFrame(matrix)
    df_mat = df_mat.ffill(axis=1).bfill(axis=1)
    matrix = df_mat.values.astype(np.float32)

    # drop VMs with too many missing bins
    enough = counts.sum(axis=1) >= min_readings
    matrix = matrix[enough]
    vm_ids_kept = [vm_ids_list[i] for i, ok in enumerate(enough) if ok]
    vmt_kept = vmt.loc[vm_ids_kept]

    print(f"[build] {len(vm_ids_kept):,} VMs have >= {min_readings} readings (>= {1-max_missing_frac:.0%} coverage).")
    if len(vm_ids_kept) == 0:
        print("[build] ERROR: 0 VMs met coverage threshold. Download more reading files or try --day-start.")
        sys.exit(1)

    # -- Weights: cores × hours (duration of window) --
    hours = (days * 86400) / 3600.0
    weights = (vmt_kept["cores"].values * hours).astype(np.float32)

    # -- Labels: map category to integer 0=Unknown, 1=Interactive, 2=Delay-insensitive --
    # Note: the real Azure V1 dataset spells it "Unkown" (sic), not "Unknown"
    label_map = {"Interactive": 1, "Delay-insensitive": 2}
    labels_str = vmt_kept["category"].values
    labels = np.array([label_map.get(s, 0) for s in labels_str], dtype=np.int8)

    return matrix, weights, labels, vm_ids_kept


def print_label_summary(labels, weights):
    total_w = weights.sum() + 1e-9
    # Note: label 0 = "Unkown" (misspelled in V1 dataset) + any other
    names = {0: "Unknown/Other", 1: "Interactive", 2: "Delay-insensitive"}
    for code, name in names.items():
        mask = labels == code
        n = mask.sum()
        w_share = weights[mask].sum() / total_w
        print(f"  {name:20s}: {n:5d} VMs  ({w_share:.1%} core-hours)")


def main():
    ap = argparse.ArgumentParser(description="Convert Azure V1 CPU trace to matrix for m0_gate_metric.py")
    ap.add_argument("--vmtable",  type=str, default=None,
                    help="Path to vmtable.csv.gz")
    ap.add_argument("--readings", type=str, required=True,
                    help="Glob pattern for vm_cpu_readings*.csv.gz, e.g. 'data/vm_cpu_readings-file-*.csv.gz'")
    ap.add_argument("--day-start", type=int, default=7,
                    help="First day of the 7-day window (0-indexed from trace start). Default 7.")
    ap.add_argument("--days",     type=int, default=7,
                    help="Window length in days. Default 7.")
    ap.add_argument("--max-vms",  type=int, default=5000,
                    help="Maximum number of VMs to include. Default 5000.")
    ap.add_argument("--max-missing", type=float, default=0.10,
                    help="Drop VMs with more than this fraction of missing bins. Default 0.10.")
    ap.add_argument("--out-trace",  type=str, default="trace.npy")
    ap.add_argument("--out-weights",type=str, default="w.npy")
    ap.add_argument("--out-labels", type=str, default="labels.npy")
    ap.add_argument("--inspect",  action="store_true",
                    help="Print inspection stats on readings files, then exit.")
    args = ap.parse_args()

    import re
    def nat_sort(file_list):
        return sorted(file_list, key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', f)])

    reading_files = nat_sort(glob.glob(args.readings))
    if not reading_files:
        print(f"ERROR: No files matched '{args.readings}'. Check the path/pattern.")
        sys.exit(1)

    if args.inspect:
        inspect_readings(args.readings)
        return

    if args.vmtable is None:
        print("ERROR: --vmtable is required for conversion (not needed for --inspect).")
        sys.exit(1)

    vmtable_df = load_vmtable(args.vmtable)

    print(f"\n[convert] {len(reading_files)} reading file(s) matched.")
    print(f"[convert] Window: day {args.day_start} to {args.day_start + args.days} "
          f"(seconds {args.day_start*86400:,} – {(args.day_start+args.days)*86400:,})")

    matrix, weights, labels, vm_ids = build_matrix(
        vmtable_df, reading_files,
        day_start=args.day_start,
        days=args.days,
        max_vms=args.max_vms,
        max_missing_frac=args.max_missing,
    )

    np.save(args.out_trace,   matrix)
    np.save(args.out_weights, weights)
    np.save(args.out_labels,  labels)

    print(f"\n[output] trace.npy  : {matrix.shape}  (float32)")
    print(f"[output] w.npy      : {weights.shape}  total core-hours={weights.sum():,.0f}")
    print(f"[output] labels.npy : {labels.shape}")
    print("\n[labels] Category breakdown by weight:")
    print_label_summary(labels, weights)

    print("\nNext step:")
    print(f"  python m0_gate_metric.py {args.out_trace} --bin-seconds 300 --weights {args.out_weights}")


if __name__ == "__main__":
    main()
