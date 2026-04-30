"""Batch cooling simulation with 4 architectural modes and time constraint."""

import numpy as np
from scipy.integrate import solve_ivp, odeint
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from model import (ProductBatch, CoolingWater, HeatExchanger, Architecture,
                   compute_heat_transfer, energy_balance_tank, calculate_LMTD)
from control import PIDState, pid_compute, reset_pid, compute_mode1_no_control


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    batch: ProductBatch
    water: CoolingWater
    exchanger: HeatExchanger

    # Time constraints
    max_batch_time: float = 3600.0  # 60 min default
    target_temp: float = 60.0       # °C

    # Mode
    mode: Architecture = Architecture.RECIRC_NO_CONTROL


@dataclass
class SimulationResults:
    """Results from simulation."""
    # Time series
    t: np.ndarray
    T_tank: np.ndarray
    T_product_out: np.ndarray
    T_water_out: np.ndarray
    Q_W: np.ndarray
    energy_J: np.ndarray

    # Flows
    water_flow_kg_s: np.ndarray
    product_flow_kg_s: np.ndarray
    product_fraction: np.ndarray

    # Time constraint
    time_to_60C_s: Optional[float] = None
    constraint_satisfied: bool = False
    constraint_violation_max_C: float = 0.0

    # Summary metrics
    total_energy_MJ: float = 0.0
    avg_power_kW: float = 0.0
    avg_water_flow_th: float = 0.0

    # Time penalty
    time_penalty_s: float = 0.0
    water_penalty_C: float = 0.0
    score: float = 0.0


class BatchCoolingSimulator:
    """Simulator for batch cooling via heat exchanger with 4 modes."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.batch = config.batch
        self.water = config.water
        self.exchanger = config.exchanger

        # PID state for controlled modes
        self.pid_water = PIDState(Kp=0.5, Ti=30.0)
        self.pid_bypass = PIDState(Kp=0.3, Ti=60.0)
        self.pid_single = PIDState(Kp=0.4, Ti=20.0)

    def reset_pids(self):
        """Reset all PID controllers."""
        self.pid_water = PIDState(Kp=0.5, Ti=30.0)
        self.pid_bypass = PIDState(Kp=0.3, Ti=60.0)
        self.pid_single = PIDState(Kp=0.4, Ti=20.0)

    def derivatives_recirc(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE for recirculation modes (Mode 1, 2, 3).

        State: [T_tank, T_water_out]
        """
        T_tank, T_water_out = y

        # Water flow (possibly controlled)
        water_flow = self.water.flow_rate

        # Product flow and fraction through exchanger
        product_flow = self.batch.recirculation_flow
        product_fraction = 1.0

        # Control logic based on mode
        if self.config.mode == Architecture.RECIRC_NO_CONTROL:
            # No control - fixed flows
            pass

        elif self.config.mode == Architecture.RECIRC_WATER_CONTROL:
            # Control water flow to maintain T_water_out <= 36°C
            error = T_water_out - self.water.max_outlet_temp
            self.pid_water.integral += error * 0.5
            self.pid_water.integral = np.clip(self.pid_water.integral, -100, 100)
            output = error + 0.02 * self.pid_water.integral
            flow_mult = np.clip(1.0 + output * 0.1, 0.5, 1.5)
            water_flow = self.water.flow_rate * flow_mult

        elif self.config.mode == Architecture.RECIRC_BYPASS:
            # Water at max, control product fraction through exchanger
            water_flow = self.water.flow_rate  # Max

            # PID control on product fraction
            setpoint = 80.0  # Target temp for bypass control
            error = T_tank - setpoint
            self.pid_bypass.integral += error * 0.2
            self.pid_bypass.integral = np.clip(self.pid_bypass.integral, -50, 50)
            output = error + 0.01 * self.pid_bypass.integral
            product_fraction = np.clip(0.5 - output * 0.1, 0.2, 1.0)

        # Heat transfer
        Q, T_product_out_calc, T_water_out_calc, _ = compute_heat_transfer(
            T_tank, self.water.inlet_temp,
            product_flow, water_flow, self.exchanger, product_fraction
        )

        # Tank energy balance: dT/dt = -Q / (m * Cp)
        dT_tank = -Q / (self.batch.mass * self.batch.Cp)

        # Water side - treat as flow-through with effective thermal mass
        # dT_water/dt proportional to Q
        C_water = water_flow * self.water.Cp
        if C_water > 0:
            dT_water = Q / C_water
        else:
            dT_water = 0.0

        return np.array([dT_tank, dT_water])

    def derivatives_single_pass(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE for single pass mode (Mode 4).

        State: [T_product_out]
        Product flows once through, no tank accumulation.
        Variable: product flow rate to maintain T_out = 60°C
        """
        T_product_out = y[0]
        T_product_in = self.batch.initial_temp

        # Product flow (possibly controlled)
        product_flow = self.batch.recirculation_flow

        # Control: adjust flow to maintain T_out = 60°C
        error = T_product_out - self.config.target_temp
        self.pid_single.integral += error * 0.5
        self.pid_single.integral = np.clip(self.pid_single.integral, -50, 50)
        output = error + 0.02 * self.pid_single.integral
        flow_mult = np.clip(1.0 + output * 0.15, 0.7, 1.3)
        product_flow = self.batch.recirculation_flow * flow_mult

        # Heat transfer
        Q, _, T_water_out_calc, _ = compute_heat_transfer(
            T_product_in, self.water.inlet_temp,
            product_flow, self.water.flow_rate, self.exchanger, 1.0
        )

        # Product outlet temp change rate
        C_product = product_flow * self.batch.Cp
        if C_product > 0:
            dT_out = -Q / C_product
        else:
            dT_out = 0.0

        return np.array([dT_out])

    def run(self) -> SimulationResults:
        """Run simulation for configured mode.

        Returns:
            SimulationResults
        """
        self.reset_pids()

        # Initial conditions
        if self.config.mode == Architecture.SINGLE_PASS:
            y0 = np.array([self.batch.initial_temp])
            max_t = 7200.0  # 2 hours max for single pass
        else:
            y0 = np.array([self.batch.initial_temp, self.water.inlet_temp])
            max_t = self.config.max_batch_time

        # Event: target temperature reached
        def event_target(t, y):
            if self.config.mode == Architecture.SINGLE_PASS:
                return y[0] - self.config.target_temp
            else:
                return y[0] - self.config.target_temp

        event_target.terminal = True
        event_target.direction = -1

        # Solve ODE
        if self.config.mode == Architecture.SINGLE_PASS:
            sol = solve_ivp(
                self.derivatives_single_pass,
                (0.0, max_t), y0,
                events=event_target,
                dense_output=True,
                max_step=1.0,
                rtol=1e-4, atol=1e-6
            )
        else:
            sol = solve_ivp(
                self.derivatives_recirc,
                (0.0, max_t), y0,
                events=event_target,
                dense_output=True,
                max_step=1.0,
                rtol=1e-4, atol=1e-6
            )

        t_eval = sol.t
        y_eval = sol.y

        n = len(t_eval)

        # Extract results
        if self.config.mode == Architecture.SINGLE_PASS:
            T_tank = np.full(n, self.batch.initial_temp)  # Inlet temp
            T_product_out = y_eval[0]
            T_water_out = np.zeros(n)
            product_fraction = np.ones(n)
            water_flow_arr = np.full(n, self.water.flow_rate)
            product_flow_arr = np.full(n, self.batch.recirculation_flow)
        else:
            T_tank = y_eval[0]
            T_product_out = T_tank  # For recirculation, outlet returns to tank
            T_water_out = y_eval[1]
            product_fraction = np.ones(n)
            water_flow_arr = np.full(n, self.water.flow_rate)
            product_flow_arr = np.full(n, self.batch.recirculation_flow)

        # Compute heat transfer for each point
        Q_arr = np.zeros(n)
        for i in range(n):
            if self.config.mode == Architecture.SINGLE_PASS:
                Q_arr[i], _, T_water_out[i], _ = compute_heat_transfer(
                    self.batch.initial_temp, self.water.inlet_temp,
                    self.batch.recirculation_flow, self.water.flow_rate,
                    self.exchanger, 1.0
                )
            else:
                Q_arr[i], _, T_water_out[i], _ = compute_heat_transfer(
                    T_tank[i], self.water.inlet_temp,
                    self.batch.recirculation_flow, self.water.flow_rate,
                    self.exchanger, 1.0
                )

        # Cumulative energy
        dt = np.diff(np.concatenate([[0], t_eval]))
        energy_J = np.cumsum(Q_arr * dt)

        # Metrics
        time_to_60C = None
        for i, T in enumerate(T_tank):
            if T <= self.config.target_temp:
                time_to_60C = t_eval[i]
                break

        max_water_temp = np.max(T_water_out) if n > 0 else 0.0
        constraint_satisfied = max_water_temp <= self.water.max_outlet_temp
        constraint_violation = max(0.0, max_water_temp - self.water.max_outlet_temp)

        total_energy = energy_J[-1] if n > 0 else 0.0
        avg_power = total_energy / t_eval[-1] if t_eval[-1] > 0 else 0.0
        avg_water_flow_kg_s = np.mean(water_flow_arr)
        avg_water_flow_th = avg_water_flow_kg_s * 3.6  # Convert to t/h

        # Scoring
        time_penalty = 0.0
        if time_to_60C is not None and time_to_60C > self.config.max_batch_time:
            time_penalty = time_to_60C - self.config.max_batch_time

        water_penalty = constraint_violation * 60  # seconds equivalent penalty

        # Score: lower is better (time + penalties)
        score = (time_to_60C if time_to_60C else max_t) + time_penalty + water_penalty

        return SimulationResults(
            t=t_eval,
            T_tank=T_tank,
            T_product_out=T_product_out,
            T_water_out=T_water_out,
            Q_W=Q_arr,
            energy_J=energy_J,
            water_flow_kg_s=water_flow_arr,
            product_flow_kg_s=product_flow_arr,
            product_fraction=product_fraction,
            time_to_60C_s=time_to_60C,
            constraint_satisfied=constraint_satisfied,
            constraint_violation_max_C=constraint_violation,
            total_energy_MJ=total_energy / 1e6,
            avg_power_kW=avg_power / 1000,
            avg_water_flow_th=avg_water_flow_th,
            time_penalty_s=time_penalty,
            water_penalty_C=water_penalty,
            score=score
        )


def run_all_modes(config: SimulationConfig) -> Dict[str, SimulationResults]:
    """Run simulation for all 4 modes.

    Returns:
        Dictionary mapping mode name to results
    """
    results = {}

    modes = [
        Architecture.RECIRC_NO_CONTROL,
        Architecture.RECIRC_WATER_CONTROL,
        Architecture.RECIRC_BYPASS,
        Architecture.SINGLE_PASS
    ]

    for mode in modes:
        cfg = SimulationConfig(
            batch=config.batch,
            water=config.water,
            exchanger=config.exchanger,
            max_batch_time=config.max_batch_time,
            target_temp=config.target_temp,
            mode=mode
        )

        sim = BatchCoolingSimulator(cfg)
        results[mode.value] = sim.run()

    return results


def create_comparison_dataframe(results: Dict[str, SimulationResults],
                                 max_batch_time: float) -> Dict:
    """Create comparison table for all modes.

    Returns:
        Dictionary with comparison metrics and scores
    """
    mode_names = {
        "mode1_no_control": "1 - Sans contrôle",
        "mode2_water_control": "2 - Contrôle débit eau",
        "mode3_bypass": "3 - Bypass produit",
        "mode4_single_pass": "4 - Passage unique"
    }

    comparison = {}

    for name, res in results.items():
        time_val = res.time_to_60C_s if res.time_to_60C_s else res.t[-1]
        time_str = f"{time_val:.0f}s" if time_val else "Non atteint"

        # Normalized scores (0-100, higher is better)
        time_score = 100 * max(0, 1 - time_val / max_batch_time) if time_val else 0
        constraint_score = 100 if res.constraint_satisfied else 0
        efficiency_score = 100 * res.avg_power_kW / 1000 if res.avg_power_kW else 0

        # Overall score (weighted)
        overall_score = 0.4 * time_score + 0.3 * constraint_score + 0.3 * efficiency_score

        comparison[name] = {
            "name": mode_names.get(name, name),
            "time_to_60C": time_str,
            "time_to_60C_s": time_val,
            "constraint_ok": "✅" if res.constraint_satisfied else f"❌ +{res.constraint_violation_max_C:.1f}°C",
            "constraint_violation_C": res.constraint_violation_max_C,
            "total_energy_MJ": f"{res.total_energy_MJ:.1f}",
            "avg_power_kW": f"{res.avg_power_kW:.1f}",
            "avg_water_flow_th": f"{res.avg_water_flow_th:.0f}",
            "time_score": time_score,
            "constraint_score": constraint_score,
            "efficiency_score": efficiency_score,
            "overall_score": overall_score
        }

    return comparison