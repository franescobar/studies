"""
Miscellaneous functions for converting units.
"""

import numpy as np


def pol2rect(mag: float, degrees: float) -> complex:
    """
    Return complex number defined by its magnitude and its angle in degrees.
    """

    return mag * np.exp(1j * np.deg2rad(degrees))


def rect2pol(z: complex) -> tuple[float, float]:
    """
    Return polar form (magnitude, angle in degrees) of complex number.
    """

    return np.abs(z), np.rad2deg(np.angle(z))


def var2mho(Mvar_3P: float, kV_LL: float) -> float:
    """
    Convert reactive power in Mvar to susceptance in mho.
    """

    return Mvar_3P / kV_LL**2


def change_base(
    quantity: complex,
    base_MVA_old: float,
    base_MVA_new: float,
    base_kV_old: float = 1.0,
    base_kV_new: float = 1.0,
    type: str = "Z",
) -> complex:
    """
    Convert quantity (impedance or power) to another base.
    """

    if type == "Z":
        Zb_old_ohm = base_kV_old**2 / base_MVA_old
        Zb_new_ohm = base_kV_new**2 / base_MVA_new
        return quantity * Zb_old_ohm / Zb_new_ohm
        # return quantity * base_MVA_new / base_MVA_old
    elif type == "Y":
        Zb_old_ohm = base_kV_old**2 / base_MVA_old
        Zb_new_ohm = base_kV_new**2 / base_MVA_new
        Yb_old_mho = 1 / Zb_old_ohm
        Yb_new_mho = 1 / Zb_new_ohm
        return quantity * Yb_old_mho / Yb_new_mho
        # return quantity * base_MVA_old / base_MVA_new
    elif type == "S":
        return quantity * base_MVA_old / base_MVA_new
    else:
        raise ValueError("type must be either 'Z' or 'S'")
