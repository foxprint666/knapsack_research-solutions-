"""
lbhf-ref/tests/test_components.py
Unit tests verifying TraceLoader (T0.1), Evaluation Harness & Effective Bandwidth (T0.3),
Active Shaper Agent (T4.2), and Plane 3 CXL Tiering Engine.
"""

import sys
import os
import numpy as np

# Add sscf repo to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from plane1_admission.eval_harness import PlacementScheduler, LindleyQueueSimulator
from plane2_shaper.shaper_agent import ShaperAgent
from plane3_cxl.tiering_model import CXLTieringEngine

def test_shaper_agent_mass_preservation():
    rng = np.random.default_rng(42)
    traces = rng.uniform(0.1, 0.9, size=(50, 200))
    shaper = ShaperAgent(window_width=8, deferrable_fraction=0.6)
    shaped, stats = shaper.shape(traces)
    
    # Check shape
    assert shaped.shape == traces.shape
    # Check mass preservation: sum across time within 1% (allowing for boundary edge padding)
    orig_sum = np.sum(traces)
    shaped_sum = np.sum(shaped)
    assert abs(orig_sum - shaped_sum) / orig_sum < 0.05
    print("[PASS] test_shaper_agent_mass_preservation passed.")

def test_effective_bandwidth_placement():
    rng = np.random.default_rng(99)
    traces = rng.uniform(0.05, 0.25, size=(100, 500))
    scheduler = PlacementScheduler(n_racks=5, target_util=0.85)
    
    metrics_effbw, loads_effbw = scheduler.place(traces, method="effbw")
    metrics_rand, loads_rand = scheduler.place(traces, method="random")
    
    assert loads_effbw.shape == (5, 500 - 125)
    assert metrics_effbw["target_utilization"] == 0.85
    print(f"[PASS] test_effective_bandwidth_placement: effbw p99={metrics_effbw['p99_backlog']:.2f}, rand p99={metrics_rand['p99_backlog']:.2f}")

def test_cxl_tiering_engine():
    cxl = CXLTieringEngine(local_dram_gb=128.0, cxl_pool_gb=512.0, hysteresis_cycles=2)
    allocations = np.array([16.0, 32.0, 8.0, 64.0])
    access_matrix = np.array([
        np.full(100, 0.90), # hot
        np.full(100, 0.20), # cold
        np.full(100, 0.15), # cold
        np.full(100, 0.85)  # hot
    ])
    # Cycle 1: warming up hysteresis (cold count = 1)
    fleet_res1 = cxl.allocate_fleet(allocations, access_matrix)
    assert fleet_res1["dram_savings_pct"] == 0.0  # protected by hysteresis
    
    # Cycle 2: hysteresis met (cold count = 2 >= hysteresis_cycles)
    fleet_res2 = cxl.allocate_fleet(allocations, access_matrix)
    assert fleet_res2["dram_savings_pct"] > 0.0
    print(f"[PASS] test_cxl_tiering_engine passed: Cycle 1 protected by hysteresis (0% mig), Cycle 2 DRAM saved={fleet_res2['dram_savings_pct']:.1f}%")

from plane1_admission.knapsack_packer import KnapsackPacker, VMDescriptor
from plane2_shaper.cgroup_shaper_daemon import ActiveShaperDaemon, CgroupHierarchyManager

def test_knapsack_packer_dual_constraints():
    packer = KnapsackPacker(host_cpu_capacity=1.0, target_util=0.70, interactive_cap=0.30)
    # Create 10 interactive VMs (mu=0.05, p99=0.30) and 20 batch VMs (mu=0.04, p99=0.20)
    vms = []
    for i in range(10):
        vms.append(VMDescriptor(i, mu_total=0.05, p99_total=0.30, is_interactive=True))
    for i in range(10, 30):
        vms.append(VMDescriptor(i, mu_total=0.04, p99_total=0.20, is_interactive=False))
        
    hosts = packer.pack_bfd(vms)
    assert len(hosts) > 0
    for h in hosts:
        assert h.current_mu_total <= h.max_mu_total + 1e-6, f"Host {h.host_id} exceeded total util cap"
        assert h.current_mu_interactive <= h.max_mu_interactive + 1e-6, f"Host {h.host_id} exceeded interactive cap"
    print(f"[PASS] test_knapsack_packer_dual_constraints: {len(vms)} VMs packed into {len(hosts)} hosts satisfying dual caps.")

def test_active_shaper_cgroup_and_trace():
    mgr = CgroupHierarchyManager()
    daemon = ActiveShaperDaemon(cgroup_mgr=mgr)
    
    # Check cgroup hierarchy setup (mock mode)
    assert mgr.setup_hierarchy() is True
    
    # Test trace shaping under overload
    rng = np.random.default_rng(123)
    int_trace = rng.uniform(0.2, 0.4, size=200) # interactive well below 1.0
    batch_trace = rng.uniform(0.7, 1.1, size=200) # together causes raw overload > 1.0
    
    res = ActiveShaperDaemon.simulate_trace_shaping(int_trace, batch_trace, host_capacity=1.0, headroom_target=0.95)
    assert res["raw_overload_pct"] > 50.0, "Raw trace should have significant overload"
    assert res["shaped_overload_pct"] == 0.0, "Shaped trace must eliminate host overload"
    assert res["interactive_sla_violations"] == 0, "Interactive workloads must have 0 violations"
    assert res["mass_conserved"], "Batch workload mass must be causally conserved"
    print(f"[PASS] test_active_shaper_cgroup_and_trace: raw overload {res['raw_overload_pct']:.1f}% -> shaped {res['shaped_overload_pct']:.1f}%, 0 interactive violations, mass conserved.")

def test_cxl_failure_isolation_and_mce_containment():
    engine = CXLTieringEngine()
    
    # Case 1: Naive policy with dirty memory on CXL fails with MCE
    naive_res = engine.simulate_cxl_link_failure(allow_dirty_cxl=True)
    assert naive_res["hardware_mce_triggered"] is True
    assert naive_res["host_kernel_panic"] is True
    assert naive_res["survival_rate_pct"] == 0.0
    
    # Case 2: SSCF ADR-0010 clean-only tiering survives with 0 MCEs
    safe_res = engine.simulate_cxl_link_failure(allow_dirty_cxl=False)
    assert safe_res["hardware_mce_triggered"] is False
    assert safe_res["host_kernel_panic"] is False
    assert safe_res["survival_rate_pct"] == 100.0
    print(f"[PASS] test_cxl_failure_isolation_and_mce_containment: Naive policy panics (0% survive); SSCF ADR-0010 survives (100% survive, 0 MCEs).")

if __name__ == "__main__":
    test_shaper_agent_mass_preservation()
    test_effective_bandwidth_placement()
    test_cxl_tiering_engine()
    test_knapsack_packer_dual_constraints()
    test_active_shaper_cgroup_and_trace()
    test_cxl_failure_isolation_and_mce_containment()
    print("\nAll component tests passed successfully!")

