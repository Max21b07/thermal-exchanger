"""Control logic for cooling system modes."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from enum import Enum


@dataclass
class PIDState:
    """PID controller state."""
    Kp: float = 1.0
    Ti: float = 100.0  # Integral time [s]
    Td: float = 0.0    # Derivative time [s]

    integral: float = 0.0
    prev_error: float = 0.0
    prev_time: float = 0.0


def pid_compute(pid: PIDState, error: float, time: float,
                output_min: float = -1e9, output_max: float = 1e9) -> float:
    """Compute PID output.

    Args:
        pid: PID state
        error: Current error (setpoint - measured)
        time: Current time [s]
        output_min: Minimum output value
        output_max: Maximum output value

    Returns:
        Control output
    """
    dt = time - pid.prev_time if pid.prev_time > 0 else 0.0

    # Proportional
    P = pid.Kp * error

    # Integral with anti-windup
    if dt > 0:
        pid.integral += error * dt

    # Clamp integral
    max_integral = 1000.0
    pid.integral = np.clip(pid.integral, -max_integral, max_integral)
    I = (pid.Kp / pid.Ti) * pid.integral if pid.Ti > 0 else 0.0

    # Derivative
    D = 0.0
    if dt > 0 and pid.Td > 0:
        D = pid.Kp * pid.Td * (error - pid.prev_error) / dt

    output = P + I + D
    output = np.clip(output, output_min, output_max)

    pid.prev_error = error
    pid.prev_time = time

    return output


def reset_pid(pid: PIDState):
    """Reset PID state."""
    pid.integral = 0.0
    pid.prev_error = 0.0
    pid.prev_time = 0.0


def compute_mode1_no_control(water_flow_nominal: float, product_flow_nominal: float,
                              **kwargs) -> Tuple[float, float]:
    """Mode 1: No control - fixed flows."""
    return water_flow_nominal, product_flow_nominal


def compute_mode2_water_control(T_water_out: float, water_flow_nominal: float,
                                 time: float, pid: PIDState,
                                 setpoint: float = 36.0) -> Tuple[float, PIDState]:
    """Mode 2: Control water flow to maintain T_water_out <= 36°C.

    Variable manipulated: water flow rate
    """
    error = T_water_out - setpoint
    output = pid_compute(pid, error, time, output_min=-0.5, output_max=0.5)

    # Modify water flow: 0.5x to 1.5x nominal
    flow_mult = 1.0 + output
    flow_mult = np.clip(flow_mult, 0.5, 1.5)

    return water_flow_nominal * flow_mult, pid


def compute_mode3_bypass_control(T_tank: float, product_flow_nominal: float,
                                  time: float, pid: PIDState) -> Tuple[float, PIDState]:
    """Mode 3: Control product fraction through exchanger.

    Water at MAX (1000 t/h assumed). PID controls product fraction.
    Objective: reach 60°C fastest.
    """
    # Target: tank temp reaching 60°C
    # When hot, full flow through exchanger. When cold, reduce.
    setpoint = 75.0  # Target tank temp for control logic

    error = T_tank - setpoint
    output = pid_compute(pid, error, time, output_min=-0.5, output_max=0.5)

    # Product fraction: 0.2 to 1.0
    frac = 0.5 - output
    frac = np.clip(frac, 0.2, 1.0)

    return frac, pid


def compute_mode4_single_pass(T_product_out: float, product_flow_nominal: float,
                               time: float, pid: PIDState,
                               setpoint: float = 60.0) -> Tuple[float, PIDState]:
    """Mode 4: Control product flow to maintain T_out = 60°C.

    Single pass - no tank accumulation.
    """
    error = T_product_out - setpoint
    output = pid_compute(pid, error, time, output_min=-0.3, output_max=0.3)

    # Flow multiplier: 0.7 to 1.3
    flow_mult = 1.0 + output
    flow_mult = np.clip(flow_mult, 0.7, 1.3)

    return product_flow_nominal * flow_mult, pid