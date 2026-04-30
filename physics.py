"""Physical models for tubular heat exchanger simulation."""

import numpy as np
from typing import Tuple


class FluidProperties:
    """Thermophysical properties of a fluid."""

    def __init__(self, name: str, rho: float, Cp: float, mu: float = None, k: float = None):
        self.name = name
        self.rho = rho          # Density [kg/m³]
        self.Cp = Cp            # Specific heat [J/(kg·K)]
        self.mu = mu            # Dynamic viscosity [Pa·s]
        self.k = k              # Thermal conductivity [W/(m·K)]


def calculate_Nu_laminar(Re: float, Pr: float, L: float, D: float) -> float:
    """Nusselt number for laminar flow in tubes (L/D > 60).
    Using Dittus-Boelter correlation for laminar flow.
    Nu = 3.66 for fully developed laminar flow (constant wall temp).
    Or Nu = 1.86 * (Re * Pr * D/L)^(1/3) for developing flow.
    """
    if L / D > 60:
        return 3.66
    else:
        # Gnielinski correlation for developing laminar flow
        if Re * Pr * D / L > 100:
            Nu = 1.86 * (Re * Pr * D / L) ** (1/3)
            return min(Nu, 3.66)
        return 3.66


def calculate_Nu_turbulent(Re: float, Pr: float, heating: bool = True) -> float:
    """Nusselt number for turbulent flow using Dittus-Boelter.
    heating=True for fluid being heated, False for fluid being cooled.
    """
    n = 0.4 if heating else 0.3
    return 0.023 * Re ** 0.8 * Pr ** n


def calculate_pressure_drop(v: float, L: float, D: float, rho: float, mu: float) -> float:
    """Calculate pressure drop using Darcy-Weisbach.
    Returns pressure drop in Pa.
    """
    Re = rho * v * D / mu
    if Re < 2300:
        f = 64 / Re
    elif Re < 4000:
        f = 0.316 / Re ** 0.25
    else:
        f = 0.184 * Re ** (-0.2)

    return f * (L / D) * (rho * v ** 2 / 2)


def calculate_heat_transfer_coeff(h_hot: float, h_cold: float, R_fouling: float = 0.0001) -> float:
    """Calculate overall heat transfer coefficient.
    1/U = 1/h_hot + R_fouling + 1/h_cold (for thin wall)
    """
    return 1.0 / (1.0 / h_hot + R_fouling + 1.0 / h_cold)


def calculate_LMTD(T_hot_in: float, T_hot_out: float, T_cold_in: float, T_cold_out: float) -> float:
    """Calculate Log Mean Temperature Difference.
    Counter-current flow assumed.
    """
    if T_hot_out == T_cold_in:
        # Co-current approximation
        dT1 = T_hot_in - T_cold_in
        dT2 = T_hot_out - T_cold_out
    else:
        # Counter-current
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in

    if abs(dT1 - dT2) < 0.01:
        return (dT1 + dT2) / 2

    return (dT1 - dT2) / np.log(dT1 / dT2) if dT1 / dT2 > 0 else (dT1 + dT2) / 2


def calculate_mean_temperature_difference(T_hot_in: float, T_hot_out: float,
                                          T_cold_in: float, T_cold_out: float) -> float:
    """Calculate arithmetic mean temperature difference (simpler approach)."""
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    return (dT1 + dT2) / 2


def calculate_Reynolds(rho: float, v: float, D: float, mu: float) -> float:
    """Calculate Reynolds number."""
    return rho * v * D / mu


def calculate_Prandtl(Cp: float, mu: float, k: float) -> float:
    """Calculate Prandtl number."""
    return Cp * mu / k