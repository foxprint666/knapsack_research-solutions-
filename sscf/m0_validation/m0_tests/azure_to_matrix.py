"""
azure_to_matrix.py - Converts Azure VM CPU readings and VM table metadata into
a clean (N_workloads x T_timebins) matrix and corresponding core-hour weights vector.

Supported inputs:
  1. Azure VM CPU readings CSV / CSV.GZ (timestamp, vm_id, min_cpu, max_cpu, avg_cpu).
  2. Azure VM Table CSV / CSV.GZ (vm_id, ..., vm_virtual_core_count_bucket).
  3. Standalone CPU readings table with core count or duration.
  4. Built-in synthetic Azure-like generator for dry runs (--generate-sample).

Usage:
  # With real Azure trace files:
  python azure_to_matrix.py --readings vm_cpu_readings.csv --vmtable vmtable.csv \
                            --output-trace trace.npy --output-weights w.npy \
                            --bin-seconds 300 --min-days 3

  # Generate a sample Azure-like dataset to verify pipeline:
  python azure_to_matrix.py --generate-sample \
                            --output-trace trace.npy --output-weights w.npy \
                            --bin-seconds 300 --min-days 7 --num-vms 200
"""

import argparse
import gzip
import os
import sys
import numpy as np
import pandas as pd

def parse_core_bucket(bucket_val):
    """
    Parses Azure's core bucket string/int into numerical core count.
    Examples: '2', '4', '8', '32', '>32' -> 32, etc.
    """
    if pd.isna(bucket_val):
        return 2.0
    val_str = str(bucket_val).strip().replace(">", "").replace("<", "")
    try:
        return float(val_str)
    except ValueError:
        return 2.0

def load_vmtable(vmtable_path):
    """Loads Azure VM table to map vm_id -> core count / category."""
    print(f"[*] Loading VM metadata table from: {vmtable_path}")
    open_fn = gzip.open if vmtable_path.endswith(".gz") else open
    # Read relevant columns
    df = pd.read_csv(vmtable_path, usecols=lambda c: any(k in c.lower() for k in ["vm_id", "core", "category"]))
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "vm_id" in cl or cl == "vmid":
            col_map[c] = "vm_id"
        elif "core" in cl:
            col_map[c] = "cores"
        elif "category" in cl:
            col_map[c] = "category"
    df = df.rename(columns=col_map)
    if "cores" in df.columns:
        df["cores"] = df["cores"].apply(parse_core_bucket)
    else:
        df["cores"] = 2.0
    vm_meta = df.set_index("vm_id")["cores"].to_dict()
    print(f"[*] Indexed metadata for {len(vm_meta)} VMs.")
    return vm_meta

def process_azure_readings(readings_path, vmtable_path=None, bin_seconds=300, min_days=3,
                           max_vms=2000, max_rows=None):
    """
    Streams and aggregates Azure CPU readings into aligned time bins per VM.
    """
    vm_cores_map = load_vmtable(vmtable_path) if vmtable_path else {}

    print(f"[*] Processing Azure CPU readings from: {readings_path}")
    bins_per_day = 86400 // bin_seconds
    min_bins = min_days * bins_per_day

    # Detect columns
    chunksize = 200_000
    reader = pd.read_csv(readings_path, chunksize=chunksize, nrows=max_rows)
    
    vm_series = {}  # vm_id -> dict of {bin_idx: avg_cpu}
    min_timestamp = None
    
    print("[*] Reading chunks and binning timestamps...")
    row_count = 0
    for chunk in reader:
        row_count += len(chunk)
        # Normalize columns
        c_map = {}
        for c in chunk.columns:
            cl = c.lower()
            if "time" in cl:
                c_map[c] = "timestamp"
            elif "vm_id" in cl or cl == "vmid":
                c_map[c] = "vm_id"
            elif "avg" in cl or "cpu" in cl:
                c_map[c] = "avg_cpu"
        chunk = chunk.rename(columns=c_map)
        
        if min_timestamp is None:
            min_timestamp = chunk["timestamp"].min()
        else:
            min_timestamp = min(min_timestamp, chunk["timestamp"].min())
            
        # Compute bin index relative to min_timestamp
        chunk["bin_idx"] = ((chunk["timestamp"] - min_timestamp) // bin_seconds).astype(int)
        
        for vm_id, grp in chunk.groupby("vm_id"):
            if vm_id not in vm_series:
                if len(vm_series) >= max_vms * 3:
                    continue  # Limit pool size for memory efficiency
                vm_series[vm_id] = {}
            for _, row in grp.iterrows():
                b = int(row["bin_idx"])
                val = float(row["avg_cpu"])
                vm_series[vm_id][b] = val
                
        if row_count % 1_000_000 == 0:
            print(f"    Processed {row_count:,} rows; tracked {len(vm_series)} unique VMs...")

    print(f"[*] Total rows processed: {row_count:,}. Evaluating VM durations...")

    # Filter VMs that span at least min_bins
    qualified_vms = []
    max_bin_overall = 0
    for vm_id, b_dict in vm_series.items():
        if not b_dict:
            continue
        min_b = min(b_dict.keys())
        max_b = max(b_dict.keys())
        span = max_b - min_b + 1
        if span >= min_bins and len(b_dict) >= min_bins * 0.70:
            qualified_vms.append((vm_id, min_b, max_b, span))
            max_bin_overall = max(max_bin_overall, max_b)

    print(f"[*] Found {len(qualified_vms)} VMs meeting minimum duration ({min_days} days = {min_bins} bins).")
    if not qualified_vms:
        raise ValueError(f"No VMs met the minimum duration requirement of {min_days} days.")

    # Select top VMs up to max_vms
    qualified_vms = qualified_vms[:max_vms]
    
    # Establish aligned global start and duration
    # Find common window
    D_bins = min_days * bins_per_day
    matrix = np.zeros((len(qualified_vms), D_bins), dtype=np.float32)
    weights = np.zeros(len(qualified_vms), dtype=np.float32)
    vm_id_list = []

    for i, (vm_id, min_b, max_b, span) in enumerate(qualified_vms):
        vm_dict = vm_series[vm_id]
        # Align from min_b to min_b + D_bins
        row = np.zeros(D_bins, dtype=np.float32)
        last_val = 0.0
        for b in range(D_bins):
            actual_b = min_b + b
            if actual_b in vm_dict:
                last_val = vm_dict[actual_b]
            row[b] = last_val
            
        matrix[i] = row
        cores = vm_cores_map.get(vm_id, 2.0)
        # Weight = cores * hours
        duration_hours = (D_bins * bin_seconds) / 3600.0
        weights[i] = cores * duration_hours
        vm_id_list.append(vm_id)

    # Filter dead/zero-variance workloads
    active = np.std(matrix, axis=1) > 0
    matrix = matrix[active]
    weights = weights[active]
    vm_id_list = [v for v, act in zip(vm_id_list, active) if act]

    print(f"[*] Final matrix constructed: {matrix.shape[0]} workloads x {matrix.shape[1]} time bins.")
    return matrix, weights, vm_id_list

def generate_sample_azure_trace(num_vms=200, days=7, bin_seconds=300, seed=42):
    """
    Generates a realistic sample Azure VM CPU trace:
      - 28% interactive (strong diurnal sinusoidal peak, 24h period)
      - 68% delay-insensitive batch (bursty, pink noise, low diurnal)
      - 4% idle/constant
      - Core-hours weights based on realistic Azure VM core buckets (1, 2, 4, 8, 16, 32, 64)
    """
    print(f"[*] Synthesizing sample Azure-like VM trace ({num_vms} VMs, {days} days)...")
    rng = np.random.default_rng(seed)
    bins_per_day = 86400 // bin_seconds
    T = days * bins_per_day
    t = np.arange(T)
    f24 = 1.0 / bins_per_day

    matrix = np.zeros((num_vms, T), dtype=np.float32)
    weights = np.zeros(num_vms, dtype=np.float32)
    core_choices = [1, 2, 4, 8, 16, 32, 64]
    core_probs = [0.15, 0.35, 0.25, 0.15, 0.05, 0.03, 0.02]

    # Azure-like category split: 28% interactive, 68% batch, 4% other
    n_interactive = int(num_vms * 0.28)
    n_batch = int(num_vms * 0.68)
    n_idle = num_vms - n_interactive - n_batch

    idx = 0
    # 1. Interactive VMs (diurnal pattern, daylight peak with random phase offset within business hours)
    for _ in range(n_interactive):
        cores = rng.choice(core_choices, p=core_probs)
        base = rng.uniform(0.10, 0.30)
        # Business hour peak with moderate jitter
        peak_hour = rng.normal(14.0, 2.0) % 24.0
        phase = -2 * np.pi * (peak_hour / 24.0)
        diurnal = 0.20 * np.sin(2 * np.pi * f24 * t + phase)
        h12 = 0.06 * np.sin(2 * np.pi * 2 * f24 * t + 2 * phase)
        noise = rng.normal(0, 0.04, size=T)
        matrix[idx] = np.clip(base + diurnal + h12 + noise, 0.01, 1.0)
        weights[idx] = cores * (T * bin_seconds / 3600.0)
        idx += 1

    # 2. Delay-insensitive batch VMs (bursty pink noise + Pareto job bursts)
    for _ in range(n_batch):
        cores = rng.choice(core_choices, p=core_probs)
        base = rng.uniform(0.05, 0.25)
        white = rng.standard_normal(T)
        fft_w = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(T)
        freqs[0] = 1.0
        pink = np.fft.irfft(fft_w / (freqs**0.5), n=T)
        pink = 0.08 * (pink / (np.std(pink) + 1e-8))
        bursts = (rng.pareto(a=2.2, size=T) - 1.0) * (rng.random(T) < 0.04) * 0.25
        matrix[idx] = np.clip(base + pink + bursts, 0.01, 1.0)
        weights[idx] = cores * (T * bin_seconds / 3600.0)
        idx += 1

    # 3. Idle VMs
    for _ in range(n_idle):
        cores = rng.choice(core_choices, p=core_probs)
        base = rng.uniform(0.01, 0.05)
        noise = rng.normal(0, 0.005, size=T)
        matrix[idx] = np.clip(base + noise, 0.001, 0.10)
        weights[idx] = cores * (T * bin_seconds / 3600.0)
        idx += 1

    print(f"[*] Generated: {n_interactive} interactive, {n_batch} batch, {n_idle} idle VMs.")
    return matrix, weights

def main():
    parser = argparse.ArgumentParser(description="Convert Azure VM CPU traces to matrix format (trace.npy & w.npy)")
    parser.add_argument("--readings", type=str, default=None, help="Path to Azure vm_cpu_readings.csv or .csv.gz")
    parser.add_argument("--vmtable", type=str, default=None, help="Path to Azure vmtable.csv or .csv.gz (for core counts)")
    parser.add_argument("--output-trace", type=str, default="trace.npy", help="Output path for trace matrix (.npy)")
    parser.add_argument("--output-weights", type=str, default="w.npy", help="Output path for weights vector (.npy)")
    parser.add_argument("--bin-seconds", type=int, default=300, help="Uniform bin length in seconds (default: 300 = 5 min)")
    parser.add_argument("--min-days", type=int, default=3, help="Minimum days duration required per VM (default: 3)")
    parser.add_argument("--max-vms", type=int, default=1000, help="Maximum number of VMs to include in matrix")
    parser.add_argument("--generate-sample", action="store_true", help="Generate realistic sample Azure-like dataset")
    parser.add_argument("--num-vms", type=int, default=200, help="Number of VMs for sample generation")
    args = parser.parse_args()

    if args.generate_sample or args.readings is None:
        if args.readings is None and not args.generate_sample:
            print("[*] No input readings specified. Defaulting to generating sample Azure trace (--generate-sample)...")
        matrix, weights = generate_sample_azure_trace(num_vms=args.num_vms, days=args.min_days + 4,
                                                     bin_seconds=args.bin_seconds)
    else:
        matrix, weights, _ = process_azure_readings(
            readings_path=args.readings,
            vmtable_path=args.vmtable,
            bin_seconds=args.bin_seconds,
            min_days=args.min_days,
            max_vms=args.max_vms
        )

    np.save(args.output_trace, matrix)
    np.save(args.output_weights, weights)
    print(f"[+] Saved trace matrix to   : {os.path.abspath(args.output_trace)} (shape: {matrix.shape})")
    print(f"[+] Saved weights vector to : {os.path.abspath(args.output_weights)} (shape: {weights.shape}, total core-hours: {weights.sum():,.1f})")
    print("\nNext step: evaluate with:")
    print(f"  python m0_gate_metric.py {args.output_trace} --bin-seconds {args.bin_seconds} --weights {args.output_weights}")

if __name__ == "__main__":
    main()
