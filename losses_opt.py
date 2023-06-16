"""
This script solves an optimization problem to minimize losses in a power
transmission system. The system is represented by the following diagram:

    V1 ========= R1 + jX1 ========= V3 ========= R2 + jX2 ========= V2

The objective is to find the values of V1 and V2 that minimize the losses
in the system, while ensuring that all voltages are within 5% of their
nominal values.

It differs from losses_newton.py in that it uses scipy.optimize.minimize
without resorting to the Newton-Raphson method. This is achieved
by passing V3 as an optimization variable.
"""

from scipy.optimize import minimize, NonlinearConstraint
import numpy as np


def losses(x: np.ndarray) -> float:
    """
    Objective function: losses in the system.
    """

    V1x, V1y, V2x, V2y, V3x, V3y = x

    V1 = V1x + 1j * V1y
    V2 = V2x + 1j * V2y
    V3 = V3x + 1j * V3y

    I13 = (V1 - V3) / (R1 + 1j * X1)
    I23 = (V2 - V3) / (R2 + 1j * X2)

    return np.abs(I13) ** 2 * R1 + np.abs(I23) ** 2 * R2


def f2(x: np.ndarray) -> np.ndarray:
    """
    Voltage constraints.
    """

    V1x, V1y, V2x, V2y, V3x, V3y = x

    V1 = V1x + 1j * V1y
    V2 = V2x + 1j * V2y
    V3 = V3x + 1j * V3y

    return np.array([np.abs(V1), np.abs(V2), np.abs(V3)])


NLC1 = NonlinearConstraint(fun=f2, lb=0.95 * np.ones(3), ub=1.05 * np.ones(3))


def f3(x: np.ndarray) -> float:
    """
    Power-flow continuity constraints.
    """

    V1x, V1y, V2x, V2y, V3x, V3y = x

    V1 = V1x + 1j * V1y
    V2 = V2x + 1j * V2y
    V3 = V3x + 1j * V3y

    I13 = (V1 - V3) / (R1 + 1j * X1)
    I23 = (V2 - V3) / (R2 + 1j * X2)

    S13 = V3 * np.conj(I13)
    S23 = V3 * np.conj(I23)

    P_into_3 = np.real(S13 + S23)
    Q_into_3 = np.imag(S13 + S23)

    return np.array([P_into_3 - P3, Q_into_3 - Q3])


NLC2 = NonlinearConstraint(fun=f3, lb=np.zeros(2), ub=np.zeros(2))

# Define constants
R1 = 0.01
X1 = 0.1
R2 = R1
X2 = X1
P3 = 5
Q3 = 0

# Solve optimization problem
res = minimize(
    fun=losses, x0=np.array([1, 0, 1, 0, 1, 0]), constraints=[NLC1, NLC2]
)

# Unpack optimization results
V1x, V1y, V2x, V2y, V3x, V3y = res.x
losses_pu = res.fun


def print_phasor(Vx: float, Vy: float) -> str:
    """
    Pretty-print a phasor.
    """

    value = Vx + 1j * Vy
    mag = np.abs(value)
    ang = np.rad2deg(np.angle(value))

    return f"{mag:.4f} pu < {ang:.2f} degrees"


print("Solution with scipy.optimize.minimize is:")
print("V1", print_phasor(V1x, V1y))
print("V2", print_phasor(V2x, V2y))
print("V3", print_phasor(V3x, V3y))
print(f"Losses are {losses_pu:.4f} pu")
