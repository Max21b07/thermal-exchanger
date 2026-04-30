"""Control logic for cooling system scenarios."""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
from enum import Enum


class ControlType(Enum):
    """Type of control strategy."""
    NONE = "none"
    WATER_FLOW = "water_flow"
    PRODUCT_BYPASS = "product_bypass"
    SINGLE_PASS_FLOW = "single_pass_flow"


@dataclass
class PIDController:
    """Simple PID controller."""
    Kp: float = 1.0
    Ti: float = 100.0  # Integral time [s]
    Td: float = 0.0    # Derivative time [s]

    setpoint: float = 0.0
    integral: float = 0.0
    prev_error: float = 0.0
    prev_time: float = 0.0

    output_min: float = 0.0
    output_max: float = 10.0

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0

    def compute(self, measured: float, time: float) -> float:
        """Compute PID output.

        Args:
            measured: Measured value
            time: Current time [s]

        Returns:
            Control output
        """
        error = self.setpoint - measured

        dt = time - self.prev_time if self.prev_time > 0 else 0.0

        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0

        self.integral += error * dt

        # Anti-windup
        max_integral = 1000.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)

        output = (self.Kp * error +
                  self.Kp / self.Ti * self.integral +
                  self.Kp * self.Td * derivative)

        output = np.clip(output, self.output_min, self.output_max)

        self.prev_error = error
        self.prev_time = time

        return output


class CoolingController:
    """Controller for cooling system scenarios."""

    def __init__(self, control_type: ControlType):
        self.control_type = control_type
        self.pid_water = PIDController(Kp=2.0, Ti=60.0, Td=0.0)
        self.pid_product = PIDController(Kp=1.0, Ti=30.0, Td=0.0)

    def reset(self):
        """Reset all controller states."""
        self.pid_water.reset()
        self.pid_product.reset()

    def compute(self, state, architecture, time: float) -> Tuple[float, float]:
        """Compute control actions.

        Args:
            state: Current simulation state
            architecture: Architecture enum
            time: Current time [s]

        Returns:
            (water_flow_multiplier, product_fraction_through)
        """
        if self.control_type == ControlType.NONE:
            return 1.0, 1.0

        elif self.control_type == ControlType.WATER_FLOW:
            # Control water outlet temperature to max allowed
            self.pid_water.setpoint = 35.5  # Slightly below max for margin
            water_multiplier = self.pid_water.compute(state.T_water_out, time)

            # Water flow can vary from 30% to 150% of nominal
            water_multiplier = np.clip(1.0 + water_multiplier * 0.1, 0.3, 1.5)
            return water_multiplier, 1.0

        elif self.control_type == ControlType.PRODUCT_BYPASS:
            # Control product temperature via bypass
            # When tank temp is high, send more product through exchanger
            self.pid_product.setpoint = 70.0  # Target temp for control

            product_fraction = self.pid_product.compute(state.T_product, time)
            product_fraction = np.clip(0.2 + product_fraction * 0.05, 0.1, 1.0)

            return 1.0, product_fraction

        elif self.control_type == ControlType.SINGLE_PASS_FLOW:
            # Control product flow to maintain outlet temp at target
            # For single pass, we control the flow rate to achieve desired exit temp
            if state.T_product_out > 65.0:
                flow_mult = 1.2  # Increase flow to reduce heat transfer
            elif state.T_product_out < 55.0:
                flow_mult = 0.8  # Decrease flow to increase heat transfer
            else:
                flow_mult = 1.0

            return 1.0, flow_mult

        return 1.0, 1.0


def apply_control_logic(state, architecture: str, time: float,
                       water_flow_nominal: float,
                       product_flow_nominal: float,
                       pid_water: PIDController = None,
                       pid_product: PIDController = None) -> Tuple[float, float]:
    """Apply control logic based on architecture.

    Args:
        state: Current state
        architecture: Architecture string
        time: Current time
        water_flow_nominal: Nominal water flow [kg/s]
        product_flow_nominal: Nominal product flow [kg/s]
        pid_water: Water temperature PID controller
        pid_product: Product temperature PID controller

    Returns:
        (water_flow_actual, product_flow_actual)
    """
    if architecture == "recirc_no_control":
        return water_flow_nominal, product_flow_nominal

    elif architecture == "recirc_water_control":
        # PID control on water outlet temperature
        if pid_water is None:
            return water_flow_nominal, product_flow_nominal

        # Target: keep water outlet below 36°C
        error = state.T_water_out - 36.0
        pid_water.integral += error * 0.1

        # Clamp integral term
        pid_water.integral = np.clip(pid_water.integral, -100, 100)

        # PID output modifies water flow
        output = (pid_water.Kp * error +
                  pid_water.Ki * pid_water.integral)

        water_mult = np.clip(1.0 + output * 0.05, 0.5, 2.0)
        return water_flow_nominal * water_mult, product_flow_nominal

    elif architecture == "recirc_bypass":
        # Bypass control - water at max, product fraction controlled
        water_mult = 1.0  # Water at maximum

        # Control product fraction based on tank temperature
        if state.T_product > 120.0:
            frac = 1.0  # Full flow through exchanger when hot
        elif state.T_product < 80.0:
            frac = 0.3  # Reduce flow when cooler
        else:
            frac = 0.6

        return water_flow_nominal * water_mult, product_flow_nominal * frac

    elif architecture == "single_pass":
        # Single pass - control product flow based on outlet temp
        if state.T_product_out > 62.0:
            # Too hot, reduce flow
            flow_mult = 0.85
        elif state.T_product_out < 58.0:
            # Too cold, increase flow
            flow_mult = 1.15
        else:
            flow_mult = 1.0

        return water_flow_nominal, product_flow_nominal * flow_mult

    return water_flow_nominal, product_flow_nominal