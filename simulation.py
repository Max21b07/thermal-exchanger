"""Batch cooling simulation with multiple architectures."""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

from model import (ProductFluid, CoolingWater, HeatExchanger, Architecture,
                   calculate_heat_transfer, energy_balance_tank, energy_balance_single_pass)
from control import ControlType, apply_control_logic


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    product: ProductFluid
    water: CoolingWater
    exchanger: HeatExchanger
    architecture: Architecture
    t_end: float = 1800.0  # 30 minutes default
    t_eval: Optional[np.ndarray] = None
    target_temp: float = 60.0  # °C


@dataclass
class SimulationResults:
    """Results from simulation."""
    t: np.ndarray
    T_product: np.ndarray      # Tank temperature or inlet for single pass
    T_product_out: np.ndarray   # Product outlet temp
    T_water_in: np.ndarray      # Water inlet temp (constant)
    T_water_out: np.ndarray     # Water outlet temp
    Q: np.ndarray               # Heat transfer rate [W]
    energy: np.ndarray          # Cumulative energy transferred [J]
    water_flow: np.ndarray      # Water flow rate [kg/s]
    product_flow: np.ndarray    # Product flow rate [kg/s]
    product_fraction: np.ndarray  # Fraction through exchanger
    constraint_satisfied: np.ndarray  # Water temp constraint met

    # Metrics
    time_to_target: Optional[float] = None
    total_energy: Optional[float] = None
    avg_power: Optional[float] = None
    max_water_temp: Optional[float] = None


class BatchCoolingSimulator:
    """Simulator for batch cooling via heat exchanger."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.product = config.product
        self.water = config.water
        self.exchanger = config.exchanger
        self.architecture = config.architecture

        # Controller state
        self.pid_water_integral = 0.0
        self.pid_water_prev_error = 0.0

    def _derivatives_recirc(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute derivatives for recirculation modes.

        State: [T_product, T_water_out]
        """
        T_product, T_water_out = y

        # Get current flows (controlled)
        water_flow = self.water.flow_rate
        product_flow = self.product.flow_rate
        product_fraction = 1.0

        # Apply control based on architecture
        if self.architecture == Architecture.RECIRC_WATER_TEMP_CONTROL:
            # Control water flow to maintain outlet temp <= 36°C
            error = T_water_out - 36.0
            self.pid_water_integral += error * 0.5
            self.pid_water_integral = np.clip(self.pid_water_integral, -50, 50)
            output = error + 0.02 * self.pid_water_integral
            water_mult = np.clip(1.0 + output * 0.1, 0.4, 1.5)
            water_flow = self.water.flow_rate * water_mult

        elif self.architecture == Architecture.RECIRC_BYPASS:
            # Bypass control - water at max, product fraction controlled
            water_flow = self.water.flow_rate  # Max water
            if T_product > 110.0:
                product_fraction = 1.0
            elif T_product < 75.0:
                product_fraction = 0.25
            else:
                product_fraction = 0.5

        elif self.architecture == Architecture.RECIRC_NO_CONTROL:
            pass  # No control

        # Heat transfer calculation
        Q, T_product_out_calc, T_water_out_calc, _ = calculate_heat_transfer(
            T_product, self.water.inlet_temp,
            self.product, self.water, self.exchanger, product_fraction
        )

        # Tank energy balance: dT/dt = -Q / (m * Cp)
        dT_product_dt = -Q / (self.product.mass * self.product.Cp)

        # Water side - simplified as flow-through (small thermal mass)
        C_water = water_flow * self.water.Cp
        dT_water_dt = Q / C_water if C_water > 0 else 0

        return np.array([dT_product_dt, dT_water_dt])

    def _derivatives_single_pass(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute derivatives for single pass mode.

        For single pass, product flows once through exchanger.
        State: [T_product_out] - outlet temp of product
        """
        T_product_in = self.product.initial_temp
        T_product_out = y[0]

        # Control product flow based on outlet temperature
        product_flow = self.product.flow_rate

        if T_product_out > 62.0:
            product_flow *= 0.9  # Reduce flow to increase residence time
        elif T_product_out < 58.0:
            product_flow *= 1.1  # Increase flow to decrease residence time

        product_flow = np.clip(product_flow, 0.1, 2.0 * self.product.flow_rate)

        # Calculate heat transfer
        Q, _, T_water_out_calc, _ = calculate_heat_transfer(
            T_product_in, self.water.inlet_temp,
            self.product, self.water, self.exchanger, 1.0
        )

        # Product outlet temp change rate
        C_product = product_flow * self.product.Cp
        dT_out_dt = -Q / C_product if C_product > 0 else 0

        return np.array([dT_out_dt])

    def run_simulation(self) -> SimulationResults:
        """Run the simulation.

        Returns:
            SimulationResults object
        """
        # Time span
        t_span = (0.0, self.config.t_end)

        # Initial conditions
        if self.architecture == Architecture.SINGLE_PASS:
            y0 = np.array([self.product.initial_temp])
        else:
            y0 = np.array([self.product.initial_temp, self.water.inlet_temp])

        # Custom event for target temperature reached
        def event(t, y):
            if self.architecture == Architecture.SINGLE_PASS:
                return y[0] - self.config.target_temp
            else:
                return y[0] - self.config.target_temp

        event.terminal = True
        event.direction = -1  # We want T to go DOWN to target

        # Solve ODE
        if self.architecture == Architecture.SINGLE_PASS:
            sol = solve_ivp(
                self._derivatives_single_pass,
                t_span, y0,
                events=event,
                dense_output=True,
                max_step=1.0
            )
        else:
            sol = solve_ivp(
                self._derivatives_recirc,
                t_span, y0,
                events=event,
                dense_output=True,
                max_step=1.0
            )

        t_eval = sol.t
        y_eval = sol.y

        # Build results
        n = len(t_eval)

        if self.architecture == Architecture.SINGLE_PASS:
            T_product = np.full(n, self.product.initial_temp)  # Inlet constant
            T_product_out = y_eval[0]
            T_water_out = np.zeros(n)
            water_flow = np.full(n, self.water.flow_rate)
            product_flow = np.full(n, self.product.flow_rate)
            product_fraction = np.ones(n)
        else:
            T_product = y_eval[0]
            T_product_out = T_product  # For recirc, outlet = tank (mixed)
            T_water_out = y_eval[1]
            water_flow = np.full(n, self.water.flow_rate)
            product_flow = np.full(n, self.product.flow_rate)
            product_fraction = np.ones(n)

        # Calculate Q for each point
        Q = np.zeros(n)
        for i in range(n):
            if self.architecture == Architecture.SINGLE_PASS:
                Q[i], _, T_water_out[i], _ = calculate_heat_transfer(
                    self.product.initial_temp, self.water.inlet_temp,
                    self.product, self.water, self.exchanger, 1.0
                )
            else:
                Q[i], _, T_water_out[i], _ = calculate_heat_transfer(
                    T_product[i], self.water.inlet_temp,
                    self.product, self.water, self.exchanger, 1.0
                )

        # Cumulative energy
        energy = np.cumsum(Q * np.diff(np.concatenate([[0], t_eval])))

        # Constraint satisfaction
        constraint_satisfied = T_water_out <= 36.0

        # Compute metrics
        time_to_target = None
        for i, T in enumerate(T_product):
            if T <= self.config.target_temp:
                time_to_target = t_eval[i]
                break

        total_energy = energy[-1] if len(energy) > 0 else 0.0
        avg_power = total_energy / t_eval[-1] if t_eval[-1] > 0 else 0.0
        max_water_temp = np.max(T_water_out) if len(T_water_out) > 0 else 0.0

        return SimulationResults(
            t=t_eval,
            T_product=T_product,
            T_product_out=T_product_out,
            T_water_in=np.full(n, self.water.inlet_temp),
            T_water_out=T_water_out,
            Q=Q,
            energy=energy,
            water_flow=water_flow,
            product_flow=product_flow,
            product_fraction=product_fraction,
            constraint_satisfied=constraint_satisfied,
            time_to_target=time_to_target,
            total_energy=total_energy,
            avg_power=avg_power,
            max_water_temp=max_water_temp
        )


class MultiScenarioComparison:
    """Compare multiple cooling scenarios."""

    def __init__(self, config_base: SimulationConfig):
        self.config_base = config_base

    def run_all_scenarios(self) -> Dict[str, SimulationResults]:
        """Run all 4 architectures.

        Returns:
            Dictionary mapping architecture name to results
        """
        results = {}

        architectures = [
            Architecture.RECIRC_NO_CONTROL,
            Architecture.RECIRC_WATER_TEMP_CONTROL,
            Architecture.RECIRC_BYPASS,
            Architecture.SINGLE_PASS
        ]

        for arch in architectures:
            config = SimulationConfig(
                product=self.config_base.product,
                water=self.config_base.water,
                exchanger=self.config_base.exchanger,
                architecture=arch,
                t_end=self.config_base.t_end,
                target_temp=self.config_base.target_temp
            )

            sim = BatchCoolingSimulator(config)
            results[arch.value] = sim.run_simulation()

        return results


def create_comparison_table(results: Dict[str, SimulationResults]) -> Dict[str, dict]:
    """Create comparison metrics for all scenarios.

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}

    for name, res in results.items():
        time_target = res.time_to_target if res.time_to_target else res.t[-1]
        time_str = f"{time_target:.0f}s" if time_target else "Non atteint"

        avg_water_flow = np.mean(res.water_flow) if len(res.water_flow) > 0 else 0

        # Proxy for operational cost: water flow + pumping energy
        pumping_proxy = avg_water_flow * 0.5  # Simplified

        comparison[name] = {
            "time_to_60°C": time_str,
            "time_to_60°C_s": res.time_to_target if res.time_to_target else float('inf'),
            "total_energy_MJ": res.total_energy / 1e6 if res.total_energy else 0,
            "avg_power_kW": res.avg_power / 1000 if res.avg_power else 0,
            "max_water_temp": f"{res.max_water_temp:.1f}°C" if res.max_water_temp else "N/A",
            "constraint_met": res.max_water_temp <= 36.0 if res.max_water_temp else False,
            "avg_water_flow_kg_s": avg_water_flow,
            "pumping_proxy": pumping_proxy
        }

    return comparison