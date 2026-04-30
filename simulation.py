"""Heat exchanger simulation engine with dynamic energy balances."""

import numpy as np
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class ControlMode(Enum):
    """Control mode for the simulation."""
    NO_CONTROL = "no_control"
    CONTROL_WATER_OUTLET = "water_outlet"
    CONTROL_PRODUCT_TEMP = "product_temp"


@dataclass
class SimulationParameters:
    """All parameters needed for simulation."""
    # Hot fluid (product recirculating)
    m_hot: float           # Mass flow rate [kg/s]
    T_hot_inlet: float     # Inlet temperature [°C]
    Cp_hot: float          # Specific heat [J/(kg·K)]

    # Cold fluid (refrigeration water)
    m_cold: float          # Mass flow rate [kg/s]
    T_cold_inlet: float    # Inlet temperature [°C]
    Cp_cold: float         # Specific heat [J/(kg·K)]

    # Heat exchanger
    U: float               # Overall heat transfer coefficient [W/(m²·K)]
    A: float               # Heat exchange surface [m²]

    # Tank (product)
    m_tank: float          # Mass of product in tank [kg]
    Cp_tank: float         # Specific heat of product [J/(kg·K)]
    T_tank_init: float     # Initial tank temperature [°C]

    # Control parameters
    control_mode: ControlMode = ControlMode.NO_CONTROL
    setpoint: float = 0.0          # Target temperature [°C]
    Kp: float = 1.0                # PID proportional gain
    Ti: float = 100.0              # PID integral time [s]
    Td: float = 0.0                # PID derivative time [s]


class HeatExchangerModel:
    """Dynamic heat exchanger with tank model.

    The system consists of:
    - A tank containing the product (hot fluid)
    - A tube-side where product circulates
    - A shell-side where refrigeration water flows
    """

    def __init__(self, params: SimulationParameters):
        self.params = params
        self.dt = 0.1  # Time step [s]

        # State variables
        self.T_tank = params.T_tank_init
        self.T_hot_out = params.T_hot_inlet  # Temperature leaving exchanger toward tank
        self.T_cold_out = params.T_cold_inlet  # Temperature leaving exchanger

        # Control state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_T_tank = params.T_tank_init

    def reset(self):
        """Reset to initial conditions."""
        self.T_tank = self.params.T_tank_init
        self.T_hot_out = self.params.T_hot_inlet
        self.T_cold_out = self.params.T_cold_inlet
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_T_tank = self.params.T_tank_init

    def calculate_heat_transfer(self, T_hot_in: float, T_cold_in: float) -> Tuple[float, float, float]:
        """Calculate heat transfer rate using LMTD method.

        Returns:
            Q: Heat transfer rate [W]
            T_hot_out: Hot fluid outlet temperature [°C]
            T_cold_out: Cold fluid outlet temperature [°C]
        """
        U = self.params.U
        A = self.params.A

        # Hot fluid enters from tank
        m_hot = self.params.m_hot
        Cp_hot = self.params.Cp_hot

        m_cold = self.params.m_cold
        Cp_cold = self.params.Cp_cold

        # Effectiveness-NTU method for better accuracy
        C_hot = m_hot * Cp_hot
        C_cold = m_cold * Cp_cold
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_ratio = C_min / C_max if C_max > 0 else 0

        # Number of transfer units
        NTU = U * A / C_min if C_min > 0 else 0

        # Effectiveness for counter-current flow
        if NTU <= 0:
            epsilon = 0
        elif C_ratio == 1:
            epsilon = NTU / (1 + NTU)
        else:
            epsilon = (1 - np.exp(-NTU * (1 - C_ratio))) / (1 - C_ratio * np.exp(-NTU * (1 - C_ratio)))

        # Maximum possible heat transfer
        dT_max = T_hot_in - self.params.T_cold_inlet
        Q_max = C_min * dT_max if dT_max > 0 else 0

        # Actual heat transfer
        Q = epsilon * Q_max

        # Outlet temperatures
        if C_hot > 0:
            T_hot_out = T_hot_in - Q / C_hot
        else:
            T_hot_out = T_hot_in

        if C_cold > 0:
            T_cold_out = self.params.T_cold_inlet + Q / C_cold
        else:
            T_cold_out = self.params.T_cold_inlet

        return Q, T_hot_out, T_cold_out

    def calculate_LMTD(self, T_hot_in: float, T_hot_out: float,
                       T_cold_in: float, T_cold_out: float) -> float:
        """Calculate LMTD for counter-current flow."""
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in

        if abs(dT1 - dT2) < 0.01 or dT1 / dT2 <= 0:
            return (dT1 + dT2) / 2

        return (dT1 - dT2) / np.log(dT1 / dT2)

    def update_control(self, Q: float, dt: float) -> float:
        """Update control action and return new mass flow rate modifier.

        Returns multiplier for cold water flow rate.
        """
        if self.params.control_mode == ControlMode.NO_CONTROL:
            return 1.0

        error = self.params.setpoint - self.T_tank

        if self.params.control_mode == ControlMode.CONTROL_PRODUCT_TEMP:
            # PID control for tank temperature
            self.integral_error += error * dt

            # Anti-windup
            max_integral = 1000
            self.integral_error = max(-max_integral, min(max_integral, self.integral_error))

            derivative = (error - self.prev_error) / dt if dt > 0 else 0

            # PID output
            output = (self.params.Kp * error +
                     self.params.Kp / self.params.Ti * self.integral_error +
                     self.params.Kp * self.params.Td * derivative)

            self.prev_error = error

            # Limit flow rate adjustment (0.5 to 2x)
            multiplier = np.clip(1.0 + output * 0.01, 0.5, 2.0)
            return multiplier

        elif self.params.control_mode == ControlMode.CONTROL_WATER_OUTLET:
            # Control cold water outlet temperature
            error_water = self.params.setpoint - self.T_cold_out

            self.integral_error += error_water * dt
            self.integral_error = max(-500, min(500, self.integral_error))

            output = (self.params.Kp * error_water +
                     self.params.Kp / self.params.Ti * self.integral_error)

            multiplier = np.clip(1.0 + output * 0.1, 0.3, 2.5)
            return multiplier

        return 1.0

    def step(self, dt: float = None) -> Dict[str, float]:
        """Perform one simulation step.

        Returns dictionary with current state values.
        """
        if dt is None:
            dt = self.dt

        # Heat exchanger inlet temperatures
        T_hot_in = self.T_tank  # Hot fluid comes from tank
        T_cold_in = self.params.T_cold_inlet  # Cold water inlet

        # Calculate heat transfer
        Q, T_hot_out, T_cold_out = self.calculate_heat_transfer(T_hot_in, T_cold_in)

        # Apply control
        flow_multiplier = self.update_control(Q, dt)

        # Update state variables
        self.T_hot_out = T_hot_out
        self.T_cold_out = T_cold_out

        # Tank energy balance
        # dT_tank/dt = (Q_in - Q_out) / (m_tank * Cp_tank)
        # Q_in = m_hot * Cp_hot * T_hot_out (from exchanger returning)
        # Actually the tank receives cooled product back, so:
        # The tank loses heat to the exchanger

        dT_tank = Q / (self.params.m_tank * self.params.Cp_tank)
        self.T_tank = self.T_tank - dT_tank * dt  # Tank temperature decreases as heat is removed

        return {
            'time': 0,  # Will be set by simulator
            'T_tank': self.T_tank,
            'T_hot_out': self.T_hot_out,
            'T_cold_out': self.T_cold_out,
            'Q': Q,
            'flow_multiplier': flow_multiplier
        }


class Simulator:
    """Simulator that runs the heat exchanger model over time."""

    def __init__(self, params: SimulationParameters):
        self.params = params
        self.model = HeatExchangerModel(params)
        self.results = {
            'time': [],
            'T_tank': [],
            'T_hot_out': [],
            'T_cold_out': [],
            'Q': [],
            'energy': [],
            'flow_multiplier': []
        }
        self.time = 0.0
        self.total_energy = 0.0
        self.dt = 0.5  # Simulation time step

    def reset(self):
        """Reset simulation to initial conditions."""
        self.model.reset()
        self.results = {k: [] for k in self.results}
        self.time = 0.0
        self.total_energy = 0.0

    def run(self, duration: float, dt: float = None, target_temp: float = None) -> Dict:
        """Run simulation for specified duration.

        Args:
            duration: Simulation duration [s]
            dt: Time step for recording results
            target_temp: Target temperature to stop at

        Returns:
            Dictionary with simulation results
        """
        if dt is None:
            dt = self.dt

        self.reset()
        t = 0.0
        n_steps = int(duration / dt)

        for i in range(n_steps):
            # Run multiple internal steps for accuracy
            internal_dt = dt / 10
            for _ in range(10):
                state = self.model.step(internal_dt)

            t += dt
            self.time = t

            # Accumulate energy
            Q = state['Q']
            self.total_energy += Q * dt

            # Store results
            self.results['time'].append(t)
            self.results['T_tank'].append(self.model.T_tank)
            self.results['T_hot_out'].append(self.model.T_hot_out)
            self.results['T_cold_out'].append(self.model.T_cold_out)
            self.results['Q'].append(Q)
            self.results['flow_multiplier'].append(state['flow_multiplier'])
            self.results['energy'].append(self.total_energy)

            # Check stop condition
            if target_temp is not None:
                if self.model.T_tank <= target_temp:
                    break

        self.results['total_energy'] = self.total_energy
        self.results['final_time'] = t
        self.results['target_reached'] = target_temp is not None and self.model.T_tank <= target_temp

        return self.results

    def run_step_by_step(self, dt: float = 1.0) -> Dict:
        """Run one time step. Use for real-time visualization."""
        state = self.model.step(dt)
        self.time += dt

        Q = state['Q']
        self.total_energy += Q * dt

        self.results['time'].append(self.time)
        self.results['T_tank'].append(self.model.T_tank)
        self.results['T_hot_out'].append(self.model.T_hot_out)
        self.results['T_cold_out'].append(self.model.T_cold_out)
        self.results['Q'].append(Q)
        self.results['flow_multiplier'].append(state['flow_multiplier'])
        self.results['energy'].append(self.total_energy)

        return self.results