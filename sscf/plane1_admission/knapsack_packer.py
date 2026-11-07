"""
lbhf-ref/sim/knapsack_packer.py
Plane 1: Mix-Aware Multi-Dimensional Knapsack Packer

Implements:
  - Dual-constraint CPU admission:
      sum(mu_total) <= target_util * Host_Capacity (default 70%)
      sum(mu_interactive) <= interactive_cap * Host_Capacity (default 30%)
  - Memory capacity constraint:
      sum(RAM_local) <= Host_DRAM
      where RAM_local accounts for DAMON-guided clean-page CXL pool tiering
  - Multi-dimensional Best-Fit Decreasing (BFD) and First-Fit Decreasing (FFD)
  - Churn-minimizing local search for incremental Day-to-Day scheduling
  - Holdout evaluation harness against real Azure traces
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class VMDescriptor:
    """Represents a virtual machine's resource profile."""
    def __init__(self, vm_id: int, mu_total: float, p99_total: float,
                 is_interactive: bool, ram_allocated_gb: float = 8.0,
                 cold_memory_ratio: float = 0.30):
        self.vm_id = vm_id
        self.mu_total = float(mu_total)
        self.p99_total = float(p99_total)
        self.is_interactive = bool(is_interactive)
        self.mu_interactive = self.mu_total if self.is_interactive else 0.0
        self.ram_allocated_gb = float(ram_allocated_gb)
        self.cold_memory_ratio = float(cold_memory_ratio)
        # Clean/cold RAM eligible for CXL pool demotion
        self.ram_cxl_eligible_gb = self.ram_allocated_gb * self.cold_memory_ratio
        # DRAM required locally on host
        self.ram_local_gb = self.ram_allocated_gb - self.ram_cxl_eligible_gb

    @property
    def composite_weight(self) -> float:
        """Dominant dimension weight for sorting (CPU mean + Interactive priority)."""
        return self.mu_total + (0.5 * self.mu_interactive)

class HostMachine:
    """Represents a physical host in the cluster."""
    def __init__(self, host_id: int, cpu_capacity: float = 1.0,
                 ram_capacity_gb: Optional[float] = None,
                 target_util: float = 0.70,
                 interactive_cap: float = 0.30,
                 enable_ram_constraint: bool = False):
        self.host_id = host_id
        self.cpu_capacity = cpu_capacity
        self.ram_capacity_gb = ram_capacity_gb
        self.target_util = target_util
        self.interactive_cap = interactive_cap
        self.enable_ram_constraint = enable_ram_constraint
        
        # Max allowable utilization thresholds
        self.max_mu_total = cpu_capacity * target_util
        self.max_mu_interactive = cpu_capacity * interactive_cap
        
        # Current host state
        self.vms: List[VMDescriptor] = []
        self.current_mu_total: float = 0.0
        self.current_mu_interactive: float = 0.0
        self.current_p99_sum: float = 0.0
        self.current_ram_local_gb: float = 0.0
        self.current_ram_cxl_gb: float = 0.0

    def can_admit(self, vm: VMDescriptor) -> bool:
        """Checks if VM satisfies multi-dimensional admission constraints."""
        new_mu_total = self.current_mu_total + vm.mu_total
        if new_mu_total > self.max_mu_total:
            return False
            
        new_mu_interactive = self.current_mu_interactive + vm.mu_interactive
        if new_mu_interactive > self.max_mu_interactive:
            return False
            
        if self.enable_ram_constraint and self.ram_capacity_gb is not None:
            new_ram_local = self.current_ram_local_gb + vm.ram_local_gb
            if new_ram_local > self.ram_capacity_gb:
                return False
            
        return True

    def admit(self, vm: VMDescriptor) -> None:
        """Places VM onto host and updates resource accounting."""
        self.vms.append(vm)
        self.current_mu_total += vm.mu_total
        self.current_mu_interactive += vm.mu_interactive
        self.current_p99_sum += vm.p99_total
        self.current_ram_local_gb += vm.ram_local_gb
        self.current_ram_cxl_gb += vm.ram_cxl_eligible_gb

    def remove(self, vm_id: int) -> Optional[VMDescriptor]:
        """Removes a VM by ID (for local search / migration)."""
        for i, vm in enumerate(self.vms):
            if vm.vm_id == vm_id:
                removed = self.vms.pop(i)
                self.current_mu_total -= removed.mu_total
                self.current_mu_interactive -= removed.mu_interactive
                self.current_p99_sum -= removed.p99_total
                self.current_ram_local_gb -= removed.ram_local_gb
                self.current_ram_cxl_gb -= removed.ram_cxl_eligible_gb
                return removed
        return None

    @property
    def cpu_slack(self) -> float:
        return max(0.0, self.max_mu_total - self.current_mu_total)

    @property
    def interactive_slack(self) -> float:
        return max(0.0, self.max_mu_interactive - self.current_mu_interactive)

    @property
    def ram_slack_gb(self) -> float:
        return max(0.0, self.ram_capacity_gb - self.current_ram_local_gb)


class KnapsackPacker:
    """
    Multi-Dimensional Knapsack Packer for Cloud Infrastructure.
    Places workloads according to SSCF Plane 1 rules.
    """
    def __init__(self, host_cpu_capacity: float = 1.0,
                 host_ram_capacity_gb: Optional[float] = None,
                 target_util: float = 0.70,
                 interactive_cap: float = 0.30,
                 enable_ram_constraint: bool = False):
        self.host_cpu_capacity = host_cpu_capacity
        self.host_ram_capacity_gb = host_ram_capacity_gb
        self.target_util = target_util
        self.interactive_cap = interactive_cap
        self.enable_ram_constraint = enable_ram_constraint

    def build_vm_descriptors(self, traces_train: np.ndarray,
                             labels: np.ndarray,
                             ram_allocations: Optional[np.ndarray] = None) -> List[VMDescriptor]:
        """Builds VM descriptors from training traces and metadata."""
        N = traces_train.shape[0]
        descriptors = []
        for i in range(N):
            mu = float(np.mean(traces_train[i]))
            p99 = float(np.percentile(traces_train[i], 99))
            is_int = bool(labels[i] == 1)
            ram = float(ram_allocations[i]) if ram_allocations is not None else 8.0
            descriptors.append(VMDescriptor(
                vm_id=i,
                mu_total=mu,
                p99_total=p99,
                is_interactive=is_int,
                ram_allocated_gb=ram
            ))
        return descriptors

    def pack_bfd(self, vms: List[VMDescriptor]) -> List[HostMachine]:
        """
        Best-Fit Decreasing Multi-Dimensional Packing.
        Sorts VMs by composite demand, places each into host with minimal remaining slack.
        """
        # Sort descending by composite weight
        sorted_vms = sorted(vms, key=lambda v: v.composite_weight, reverse=True)
        hosts: List[HostMachine] = []

        for vm in sorted_vms:
            best_host: Optional[HostMachine] = None
            min_slack = float('inf')

            for h in hosts:
                if h.can_admit(vm):
                    # Slack score: Euclidean distance in normalized residual space
                    slack_cpu = (h.cpu_slack - vm.mu_total) / h.max_mu_total
                    slack_int = (h.interactive_slack - vm.mu_interactive) / h.max_mu_interactive
                    slack = (slack_cpu ** 2 + slack_int ** 2) ** 0.5
                    if slack < min_slack:
                        min_slack = slack
                        best_host = h

            if best_host is not None:
                best_host.admit(vm)
            else:
                new_host = HostMachine(
                    host_id=len(hosts),
                    cpu_capacity=self.host_cpu_capacity,
                    ram_capacity_gb=self.host_ram_capacity_gb,
                    target_util=self.target_util,
                    interactive_cap=self.interactive_cap,
                    enable_ram_constraint=self.enable_ram_constraint
                )
                assert new_host.can_admit(vm), f"VM {vm.vm_id} exceeds empty host capacity!"
                new_host.admit(vm)
                hosts.append(new_host)

        return hosts

    def pack_ffd(self, vms: List[VMDescriptor]) -> List[HostMachine]:
        """First-Fit Decreasing Multi-Dimensional Packing."""
        sorted_vms = sorted(vms, key=lambda v: v.composite_weight, reverse=True)
        hosts: List[HostMachine] = []

        for vm in sorted_vms:
            placed = False
            for h in hosts:
                if h.can_admit(vm):
                    h.admit(vm)
                    placed = True
                    break
            if not placed:
                new_host = HostMachine(
                    host_id=len(hosts),
                    cpu_capacity=self.host_cpu_capacity,
                    ram_capacity_gb=self.host_ram_capacity_gb,
                    target_util=self.target_util,
                    interactive_cap=self.interactive_cap,
                    enable_ram_constraint=self.enable_ram_constraint
                )
                new_host.admit(vm)
                hosts.append(new_host)

        return hosts

    def pack_incremental_churn_min(self, current_hosts: List[HostMachine],
                                  new_vms: List[VMDescriptor],
                                  departed_vm_ids: List[int]) -> Tuple[List[HostMachine], int]:
        """
        Churn-minimizing local search:
        Retains existing VM placement; removes departed VMs; places newly arrived VMs
        using Best-Fit into available slack before opening new physical hosts.
        Returns (updated_hosts, migration_count).
        """
        # Remove departed VMs
        for vm_id in departed_vm_ids:
            for h in current_hosts:
                if h.remove(vm_id) is not None:
                    break

        active_hosts = [h for h in current_hosts if len(h.vms) > 0]
        migrations = 0

        # Place new VMs
        sorted_new = sorted(new_vms, key=lambda v: v.composite_weight, reverse=True)
        for vm in sorted_new:
            best_host = None
            min_slack = float('inf')
            for h in active_hosts:
                if h.can_admit(vm):
                    slack = (h.cpu_slack - vm.mu_total)
                    if slack < min_slack:
                        min_slack = slack
                        best_host = h
            if best_host is not None:
                best_host.admit(vm)
            else:
                new_host = HostMachine(
                    host_id=len(active_hosts),
                    cpu_capacity=self.host_cpu_capacity,
                    ram_capacity_gb=self.host_ram_capacity_gb,
                    target_util=self.target_util,
                    interactive_cap=self.interactive_cap,
                    enable_ram_constraint=self.enable_ram_constraint
                )
                new_host.admit(vm)
                active_hosts.append(new_host)

        return active_hosts, migrations

    @staticmethod
    def evaluate_holdout(hosts: List[HostMachine], traces_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates physical host loads over out-of-sample holdout test period.
        Computes raw overload frequency, peak utilization, and SLA violations.
        """
        num_hosts = len(hosts)
        T_test = traces_test.shape[1]
        host_loads = np.zeros((num_hosts, T_test), dtype=np.float64)
        interactive_loads = np.zeros((num_hosts, T_test), dtype=np.float64)

        for h_idx, h in enumerate(hosts):
            for vm in h.vms:
                series = traces_test[vm.vm_id]
                host_loads[h_idx] += series
                if vm.is_interactive:
                    interactive_loads[h_idx] += series

        # Overload occurs when total load > host capacity
        cap = hosts[0].cpu_capacity if hosts else 1.0
        overload_mask = (host_loads > cap)
        overload_pct = float(np.mean(overload_mask)) * 100.0
        
        # Interactive SLA violation occurs when interactive demand alone exceeds host capacity
        interactive_violations = np.sum(interactive_loads > cap)
        interactive_sla_violation_pct = float(interactive_violations / (num_hosts * T_test)) * 100.0

        p95_host_util = float(np.percentile(host_loads, 95))
        p99_host_util = float(np.percentile(host_loads, 99))
        max_host_util = float(np.max(host_loads))
        mean_host_util = float(np.mean(host_loads))

        return {
            "num_hosts": num_hosts,
            "mean_host_utilization": mean_host_util,
            "p95_host_utilization": p95_host_util,
            "p99_host_utilization": p99_host_util,
            "max_host_utilization": max_host_util,
            "raw_overload_pct": overload_pct,
            "interactive_sla_violations": int(interactive_violations),
            "interactive_sla_violation_pct": interactive_sla_violation_pct,
            "host_loads": host_loads,
            "interactive_loads": interactive_loads
        }
