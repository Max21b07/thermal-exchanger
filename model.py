"""Physical models for industrial batch cooling simulation."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class Architecture(Enum):
    """Cooling system architecture."""
    RECIRC_NO_CONTROL = "recirc_no_control"
    RECIRC_WATER_TEMP_CONTROL = "recirc_water_control"
    RECIRC_BYPASS = "recirc_bypass"
    SINGLE_PASS = "single_pass"


@dataclass
class ProductFluid:
    """Hot product (batch) properties."""
    mass: float           # kg
    Cp: float             # J/(kg·K)
    density: float        # kg/m³
    initial_temp: float   # °C
    flow_rate: float      # kg/s (recirculation or single pass)


@dataclass
class CoolingWater:
    """Cooling water properties."""
    flow_rate: float      # kg/s
    inlet_temp: float     # °C
    max_outlet_temp: float # °C (constraint)
    Cp: float = 4184.0    # J/(kg·K) - water


@dataclass
class HeatExchanger:
    """Heat exchanger parameters."""
    area: float           # m²
    U: float              # W/(m²·K)
    fouling_factor: float = 0.0001  # m²·K/W


@dataclass
class SimulationState:
    """Current state of the simulation."""
    time: float = 0.0

    # Product/Cuvestate
    T_product: float = 150.0      # °C (tank temperature for recirculation)
    T_product_out: float = 150.0  # °C (after heat exchanger)
    mass_product_in_exchanger: float = 1.0  # fraction through exchanger

    # Water state
    T_water_out: float = 28.0    # °C

    # Heat transfer
    Q: float = 0.0                # W
    energy_transferred: float = 0.0  # J

    # Control
    water_flow_multiplier: float = 1.0
    product_flow_multiplier: float = 1.0


def calculate_LMTD(T_hot_in: float, T_hot_out: float,
                   T_cold_in: float, T_cold_out: float) -> float:
    """Calculate Log Mean Temperature Difference.

    Args:
        T_hot_in: Hot fluid inlet temperature
        T_hot_out: Hot fluid outlet temperature
        T_cold_in: Cold fluid inlet temperature
        T_cold_out: Cold fluid outlet temperature

    Returns:
        LMTD in °C
    """
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in

    if dT1 <= 0 or dT2 <= 0:
        return (dT1 + dT2) / 2

    if abs(dT1 - dT2) < 0.1:
        return (dT1 + dT2) / 2

    return (dT1 - dT2) / np.log(dT1 / dT2)


def calculate_effectiveness_NTU(C_hot: float, C_cold: float,
                                 U: float, A: float) -> float:
    """Calculate heat exchanger effectiveness using NTU method.

    Args:
        C_hot: Heat capacity rate of hot fluid (W/K)
        C_cold: Heat capacity rate of cold fluid (W/K)
        U: Overall heat transfer coefficient (W/m²K)
        A: Heat exchange area (m²)

    Returns:
        Effectiveness (0 to 1)
    """
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)

    if C_min <= 0:
        return 0.0

    C_ratio = C_min / C_max
    NTU = U * A / C_min

    if NTU <= 0:
        return 0.0

    if C_ratio == 0:
        return 1.0 - np.exp(-NTU)

    if C_ratio == 1.0:
        return NTU / (1.0 + NTU)

    return (1.0 - np.exp(-NTU * (1.0 - C_ratio))) / (1.0 - C_ratio * np.exp(-NTU * (1.0 - C_ratio)))


def calculate_heat_transfer(T_product_in: float, T_water_in: float,
                           product: ProductFluid, water: CoolingWater,
                           exchanger: HeatExchanger,
                           fraction_through: float = 1.0) -> Tuple[float, float, float, float]:
    """Calculate heat transfer in the exchanger.

    Returns:
        Q: Heat transfer rate [W]
        T_product_out: Product outlet temperature [°C]
        T_water_out: Water outlet temperature [°C]
        LMTD: Log mean temperature difference [°C]
    """
    if fraction_through <= 0:
        return 0.0, T_product_in, T_water_in, 0.0

    m_product = product.flow_rate * fraction_through
    C_product = m_product * product.Cp
    C_water = water.flow_rate * water.Cp

    C_min = min(C_product, C_water)

    dT_max = T_product_in - T_water_in
    if dT_max <= 0:
        return 0.0, T_product_in, T_water_in, 0.0

    epsilon = calculate_effectiveness_NTU(C_product, C_water, exchanger.U, exchanger.area)

    Q_max = C_min * dT_max
    Q = epsilon * Q_max

    T_product_out = T_product_in - Q / C_product if C_product > 0 else T_product_in
    T_water_out = T_water_in + Q / C_water if C_water > 0 else T_water_in

    LMTD = calculate_LMTD(T_product_in, T_product_out, T_water_in, T_water_out)

    return Q, T_product_out, T_water_out, LMTD


def energy_balance_tank(T_product: float, Q: float, m_product: float,
                        Cp_product: float, dt: float) -> float:
    """Calculate new tank temperature after dt using energy balance.

    dT/dt = -Q / (m * Cp)

    Args:
        T_product: Current tank temperature [°C]
        Q: Heat transfer rate [W]
        m_product: Product mass in tank [kg]
        Cp_product: Specific heat [J/(kg·K)]
        dt: Time step [s]

    Returns:
        New tank temperature [°C]
    """
    if m_product <= 0 or Cp_product <= 0:
        return T_product

    dT = -Q * dt / (m_product * Cp_product)
    return T_product + dT


def energy_balance_single_pass(T_in: float, Q: float, m: float, Cp: float) -> float:
    """Calculate outlet temperature for single pass (no tank accumulation).

    Args:
        T_in: Inlet temperature [°C]
        Q: Heat transfer rate [W]
        m: Mass flow rate [kg/s]
        Cp: Specific heat [J/(kg·K)]

    Returns:
        Outlet temperature [°C]
    """
    if m <= 0 or Cp <= 0:
        return T_in

    return T_in - Q / (m * Cp)


def estimate_U_from_correlations(v_product: float, v_water: float,
                                  D_hydraulic: float,
                                  mu_product: float, mu_water: float,
                                  Cp_product: float, Cp_water: float,
                                  k_product: float, k_water: float) -> float:
    """Estimate U from flow correlations (optional).

    This is a simplified estimation. In practice, U is often measured.

    Returns:
        U in W/(m²·K)
    """
    Re_product = 0.0
    Re_water = 0.0

    if D_hydraulic > 0 and mu_product > 0:
        Re_product = 1000.0  # Assuming laminar for viscous product

    if D_hydraulic > 0 and mu_water > 0:
        Re_water = 5000.0  # Assuming turbulent for water

    # Simplified: assume typical U values for shell-and-tube
    # Product side (viscous): h ~ 100-300 W/m²K
    # Water side: h ~ 500-2000 W/m²K
    h_product = 200.0
    h_water = 1000.0

    R_fouling = 0.0001  # m²K/W

    U = 1.0 / (1.0/h_product + R_fouling + 1.0/h_water)
    return U