"""
tests/test_sscf_end_to_end.py
End-to-End System Integration Test for Stochastic-Shaped CXL Fabric (SSCF)

Integrates:
  - Plane 1: Multi-Dimensional Knapsack Packer (Admission & Placement)
  - Plane 2: Linux Host Active Shaper Daemon (Runtime Burst Management)
  - Plane 3: Failure-Isolated CXL Tiering Engine (Hardware Isolation)

Validates against real 3-day Azure fleet trace (4,994 VMs x 864 bins):
  1. Host reduction >= 23% vs Static P99 reservation baseline
  2. Next-day raw host overload < 0.5% (Day 1 & Day 2 out-of-sample)
  3. Interactive SLA violations = 0.0000%
  4. Active shaping eliminates residual tail overload while conserving batch work
  5. CXL link drop produces ZERO Machine Check Exceptions (MCE) and 100% host survival
"""

import os
import sys
import numpy as np

# Setup import paths
SSCF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SSCF_ROOT not in sys.path:
    sys.path.insert(0, SSCF_ROOT)

M0_DIR = os.path.join(SSCF_ROOT, "m0_validation")

from plane1_admission.knapsack_packer import KnapsackPacker, VMDescriptor
from plane2_shaper.cgroup_shaper_daemon import ActiveShaperDaemon, CgroupHierarchyManager
from plane3_cxl.tiering_model import CXLTieringEngine

def run_end_to_end_validation():
    print("======================================================================")
    print("   STOCHASTIC-SHAPED CXL FABRIC (SSCF) END-TO-END VALIDATION SUITE")
    print("======================================================================\n")

    # 1. Load Real Azure Fleet Trace
    trace_path = os.path.join(M0_DIR, "trace.npy")
    labels_path = os.path.join(M0_DIR, "labels.npy")
    
    if not os.path.exists(trace_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing Azure trace files in {M0_DIR}")
        
    trace = np.load(trace_path)
    labels = np.load(labels_path)
    N, T = trace.shape
    print(f"[TRACE] Loaded Azure fleet trace: {N} VMs, {T} bins (5-min intervals, 72 hours).")
    
    # 2. Partition 3-Day Holdout
    # Day 0: bins 0..288 (train)
    # Day 1: bins 288..576 (holdout 1)
    # Day 2: bins 576..864 (holdout 2)
    day0 = trace[:, 0:288]
    day1 = trace[:, 288:576]
    day2 = trace[:, 576:864]
    
    # Day 0 Statistics
    mu_day0 = np.mean(day0, axis=1)
    p99_day0 = np.percentile(day0, 99, axis=1)
    
    int_mask = (labels == 1)
    batch_mask = ~int_mask
    print(f"[DATASET] Composition: {np.sum(int_mask)} Interactive VMs ({np.sum(mu_day0[int_mask])/np.sum(mu_day0)*100:.1f}% core-hours), "
          f"{np.sum(batch_mask)} Delay-Insensitive VMs ({np.sum(mu_day0[batch_mask])/np.sum(mu_day0)*100:.1f}% core-hours).")
          
    # ------------------------------------------------------------------
    # PLANE 1 VALIDATION: Multi-Dimensional Knapsack Packing
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("   PLANE 1: Multi-Dimensional Knapsack Placement & Sizing")
    print("----------------------------------------------------------------------")
    
    # Physical Host Capacity: 16 cores @ 100% per core = 1600.0% CPU
    HOST_CAPACITY = 1600.0
    
    # Baseline 1: Static P99 Reservation
    # Total hosts needed if reserving sum(p99) <= HOST_CAPACITY per host
    p99_total = np.sum(p99_day0)
    p99_hosts_ideal = p99_total / HOST_CAPACITY
    print(f"[BASELINE] Static P99 Total Demand: {p99_total:.2f} CPU% (Ideal={p99_hosts_ideal:.1f} hosts @ {HOST_CAPACITY}%)")
    
    # SSCF Knapsack Packer
    packer = KnapsackPacker(host_cpu_capacity=HOST_CAPACITY, target_util=0.70, interactive_cap=0.30)
    vms = packer.build_vm_descriptors(day0, labels)
    hosts = packer.pack_bfd(vms)
    num_sscf_hosts = len(hosts)
    
    host_savings_pct = ((p99_hosts_ideal - num_sscf_hosts) / p99_hosts_ideal) * 100.0
    print(f"[PLANE 1] SSCF Multi-Dimensional Knapsack Hosts: {num_sscf_hosts}")
    print(f"[PLANE 1] Host Reduction vs Static P99: +{host_savings_pct:.2f}%")
    assert host_savings_pct >= 23.0, f"Host savings (+{host_savings_pct:.1f}%) fell below 23% gate!"

    # Evaluate Holdout on Day 1 (Next-Day Out-of-Sample)
    res_day1 = KnapsackPacker.evaluate_holdout(hosts, day1)
    print(f"[PLANE 1 - DAY 1] Mean Host Util: {res_day1['mean_host_utilization']/HOST_CAPACITY*100:.2f}% | "
          f"P95: {res_day1['p95_host_utilization']/HOST_CAPACITY*100:.2f}% | P99: {res_day1['p99_host_utilization']/HOST_CAPACITY*100:.2f}%")
    print(f"[PLANE 1 - DAY 1] Raw Host Overload: {res_day1['raw_overload_pct']:.4f}% | "
          f"Interactive SLA Violations: {res_day1['interactive_sla_violations']} ({res_day1['interactive_sla_violation_pct']:.4f}%)")
    assert res_day1["raw_overload_pct"] < 0.50, f"Day 1 overload ({res_day1['raw_overload_pct']:.2f}%) exceeded 0.5% gate!"
    assert res_day1["interactive_sla_violations"] == 0, "Day 1 suffered Interactive SLA violations!"

    # Evaluate Holdout on Day 2 (Two Days Out-of-Sample)
    res_day2 = KnapsackPacker.evaluate_holdout(hosts, day2)
    print(f"[PLANE 1 - DAY 2] Mean Host Util: {res_day2['mean_host_utilization']/HOST_CAPACITY*100:.2f}% | "
          f"P95: {res_day2['p95_host_utilization']/HOST_CAPACITY*100:.2f}% | P99: {res_day2['p99_host_utilization']/HOST_CAPACITY*100:.2f}%")
    print(f"[PLANE 1 - DAY 2] Raw Host Overload: {res_day2['raw_overload_pct']:.4f}% | "
          f"Interactive SLA Violations: {res_day2['interactive_sla_violations']} ({res_day2['interactive_sla_violation_pct']:.4f}%)")
    assert res_day2["raw_overload_pct"] < 0.50, f"Day 2 overload ({res_day2['raw_overload_pct']:.2f}%) exceeded 0.5% gate!"
    assert res_day2["interactive_sla_violations"] == 0, "Day 2 suffered Interactive SLA violations!"

    # ------------------------------------------------------------------
    # PLANE 2 VALIDATION: Runtime Active Shaper Daemon
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("   PLANE 2: Linux Host Active Shaper Daemon (Burst Absorption)")
    print("----------------------------------------------------------------------")
    
    # Test on the host with highest peak burst load
    peak_host_idx = int(np.argmax([np.max(res_day2["host_loads"][h_i]) for h_i in range(num_sscf_hosts)]))
    target_host = hosts[peak_host_idx]
    
    int_trace = np.zeros(day2.shape[1], dtype=np.float64)
    batch_trace = np.zeros(day2.shape[1], dtype=np.float64)
    for vm in target_host.vms:
        if vm.is_interactive:
            int_trace += day2[vm.vm_id]
        else:
            batch_trace += day2[vm.vm_id]
            
    shaper_res = ActiveShaperDaemon.simulate_trace_shaping(int_trace, batch_trace, host_capacity=HOST_CAPACITY, headroom_target=0.95)
    print(f"[PLANE 2] Peak Host #{peak_host_idx}: Raw Overload = {shaper_res['raw_overload_pct']:.2f}% -> "
          f"Shaped Overload = {shaper_res['shaped_overload_pct']:.2f}%")
    print(f"[PLANE 2] Throttled Bins: {shaper_res['throttled_bins_pct']:.2f}% | "
          f"P99 Queue Backlog: {shaper_res['p99_queue_backlog']:.4f} load-units")
    print(f"[PLANE 2] Interactive SLA Violations: {shaper_res['interactive_sla_violations']} | "
          f"Work Mass Conserved: {shaper_res['mass_conserved']}")
          
    assert shaper_res["shaped_overload_pct"] == 0.0, "Active shaper failed to eliminate host overload!"
    assert shaper_res["interactive_sla_violations"] == 0, "Active shaper compromised interactive SLA!"
    assert shaper_res["mass_conserved"], "Active shaper lost batch workload mass!"

    # ------------------------------------------------------------------
    # PLANE 3 VALIDATION: Failure-Isolated CXL Pooled Memory Tiering
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("   PLANE 3: Failure-Isolated CXL Tiering & MCE Containment")
    print("----------------------------------------------------------------------")
    
    cxl_engine = CXLTieringEngine(local_dram_gb=128.0, cxl_pool_gb=512.0, hysteresis_cycles=2)
    vm_allocs = np.array([vm.ram_allocated_gb for vm in target_host.vms])
    
    # Construct access frequencies across 3 cycles
    rng = np.random.default_rng(42)
    acc_matrix_cold = rng.uniform(0.1, 0.4, size=(len(vm_allocs), 100))
    
    # Cycle 1 (warming up hysteresis)
    c1 = cxl_engine.allocate_fleet(vm_allocs, acc_matrix_cold)
    assert c1["dram_savings_pct"] == 0.0, "Hysteresis failed to block immediate migration on cycle 1"
    
    # Cycle 2 (hysteresis met -> clean pages demoted)
    c2 = cxl_engine.allocate_fleet(vm_allocs, acc_matrix_cold)
    print(f"[PLANE 3] Host Memory Tiering: Total Allocated = {c2['total_fleet_memory_gb']:.1f} GB | "
          f"Local DRAM Used = {c2['local_dram_used_gb']:.1f} GB | CXL Demoted = {c2['cxl_pool_used_gb']:.1f} GB")
    print(f"[PLANE 3] DRAM Headroom Savings: +{c2['dram_savings_pct']:.1f}%")
    assert c2["dram_savings_pct"] > 0.0, "CXL tiering failed to achieve DRAM savings!"

    # Fault Injection: CXL Fabric Link Disconnect
    print("\n[FAULT INJECTION] Simulating sudden CXL fabric link-drop...")
    # 1. Naive Dirty Tiering (Violation of ADR-0010)
    naive_fault = cxl_engine.simulate_cxl_link_failure(allow_dirty_cxl=True)
    print(f"  [NAIVE DIRTY TIERING] Hardware MCE: {naive_fault['hardware_mce_triggered']} | "
          f"Host Kernel Panic: {naive_fault['host_kernel_panic']} | Survival: {naive_fault['survival_rate_pct']:.1f}%")
    assert naive_fault["host_kernel_panic"] is True, "Naive dirty tiering should crash on link drop!"

    # 2. SSCF Clean-Only Tiering (ADR-0010 Compliant)
    sscf_fault = cxl_engine.simulate_cxl_link_failure(allow_dirty_cxl=False)
    print(f"  [SSCF ADR-0010 CLEAN] Hardware MCE: {sscf_fault['hardware_mce_triggered']} | "
          f"Host Kernel Panic: {sscf_fault['host_kernel_panic']} | Survival: {sscf_fault['survival_rate_pct']:.1f}%")
    print(f"  Result: {sscf_fault['message']}")
    assert sscf_fault["hardware_mce_triggered"] is False, "SSCF must prevent hardware MCEs!"
    assert sscf_fault["host_kernel_panic"] is False, "SSCF must prevent host kernel panics!"
    assert sscf_fault["survival_rate_pct"] == 100.0, "SSCF must guarantee 100% host survival!"

    print("\n======================================================================")
    print("   ALL SSCF END-TO-END VALIDATION GATES PASSED (100% SUCCESS)!")
    print("======================================================================\n")

if __name__ == "__main__":
    run_end_to_end_validation()
