"""Physical models for industrial batch cooling simulation."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class Architecture(Enum):
    """Cooling system architecture."""
    RECIRC_NO_CONTROL = "mode1_no_control"
    RECIRC_WATER_CONTROL = "mode2_water_control"
    RECIRC_BYPASS = "mode3_bypass"
    SINGLE_PASS = "mode4_single_pass"


@dataclass
class ProductBatch:
    """Hot product batch properties."""
    mass: float           # kg (batch size, 45000 typical)
    Cp: float             # J/(kg·K)
    density: float        # kg/m³
    initial_temp: float   # °C
    recirculation_flow: float  # kg/s (≈ 50 kg/s for 180 t/h)


@dataclass
class CoolingWater:
    """Cooling water utility."""
    flow_rate: float      # kg/s (configurable, 400 t/h default)
    inlet_temp: float     # °C (28°C fixed)
    max_outlet_temp: float # °C (constraint: 36°C)
    Cp: float = 4184.0    # J/(kg·K) - water


@dataclass
class HeatExchanger:
    """Heat exchanger parameters."""
    area: float           # m² (80-150 m² typical)
    U: float              # W/(m²·K) (400-800 typical)
    fouling: float = 0.0001  # m²·K/W


@dataclass
class SimulationResults:
    """Results from simulation."""
    t: np.ndarray
    T_tank: np.ndarray          # Tank temperature (or product inlet for single pass)
    T_product_out: np.ndarray   # Product outlet from exchanger
    T_water_out: np.ndarray     # Water outlet temp
    Q: np.ndarray               # Heat transfer rate [W]
    energy_cumulative: np.ndarray # Cumulative energy [J]
    water_flow: np.ndarray       # Actual water flow [kg/s]
    product_flow: np.ndarray    # Actual product flow [kg/s]
    product_fraction: np.ndarray # Fraction through exchanger

    # Metrics
    time_to_60C: Optional[float] = None
    max_water_temp: Optional[float] = None
    constraint_satisfied: bool = False
    total_energy_MJ: float = 0.0
    avg_power_kW: float = 0.0


def calculate_LMTD(T_hot_in: float, T_hot_out: float,
                   T_cold_in: float, T_cold_out: float) -> float:
    """Log Mean Temperature Difference for counter-current flow."""
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in

    if dT1 <= 0 or dT2 <= 0 or abs(dT1 - dT2) < 0.1:
        return max(dT1, dT2) / 2 + min(dT1, dT2) / 2

    return (dT1 - dT2) / np.log(dT1 / dT2)


def calculate_NTU_effectiveness(C_hot: float, C_cold: float, U: float, A: float) -> float:
    """Heat exchanger effectiveness using NTU method."""
    if C_hot <= 0 or C_cold <= 0:
        return 0.0

    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_ratio = C_min / C_max

    NTU = U * A / C_min
    if NTU <= 0:
        return 0.0

    if C_ratio == 0:
        return 1.0 - np.exp(-NTU)
    elif C_ratio == 1.0:
        return NTU / (1.0 + NTU)
    else:
        numerator = 1.0 - np.exp(-NTU * (1.0 - C_ratio))
        denominator = 1.0 - C_ratio * np.exp(-NTU * (1.0 - C_ratio))
        return numerator / denominator


def compute_heat_transfer(T_product_in: float, T_water_in: float,
                         product_flow: float, water_flow: float,
                         exchanger: HeatExchanger,
                         product_fraction: float = 1.0) -> Tuple[float, float, float, float]:
    """Compute heat transfer in exchanger.

    Returns:
        Q: Heat transfer rate [W]
        T_product_out: Product outlet temp [°C]
        T_water_out: Water outlet temp [°C]
        LMTD: Log mean temp difference [°C]
    """
    if product_fraction <= 0:
        return 0.0, T_product_in, T_water_in, 0.0

    effective_product_flow = product_flow * product_fraction

    C_product = effective_product_flow * 2200.0  # Use actual Cp from batch
    C_water = water_flow * 4184.0

    dT_max = T_product_in - T_water_in
    if dT_max <= 0:
        return 0.0, T_product_in, T_water_in, 0.0

    epsilon = calculate_NTU_effectiveness(C_product, C_water, exchanger.U, exchanger.area)
    Q = epsilon * min(C_product, C_water) * dT_max

    if C_product > 0:
        T_product_out = T_product_in - Q / C_product
    else:
        T_product_out = T_product_in

    if C_water > 0:
        T_water_out = T_water_in + Q / C_water
    else:
        T_water_out = T_water_in

    LMTD = calculate_LMTD(T_product_in, T_product_out, T_water_in, T_water_out)

    return Q, T_product_out, T_water_out, LMTD


def energy_balance_tank(T_tank: float, Q: float, m_tank: float,
                        Cp: float, dt: float) -> float:
    """Tank temperature change from energy balance.

    dT/dt = -Q / (m * Cp)
    """
    if m_tank <= 0 or Cp <= 0:
        return T_tank

    dT = -Q * dt / (m_tank * Cp)
    return T_tank + dT