"""
This script solves an optimization problem to minimize losses in a power
transmission system. The system is represented by the following diagram:

    V1 ========= R1 + jX1 ========= V3 ========= R2 + jX2 ========= V2

The objective is to find the values of V1 and V2 that minimize the losses
in the system, while ensuring that all voltages are within 5% of their
nominal values.
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


def jac(f: callable, x: np.ndarray, h: float = 1e-9) -> np.ndarray:
    """
    Compute numerical Jacobian of f at x, using a perturbation h.

    x must be a vector column (2D).
    """

    # Initialize J
    N = x.shape[0]
    J = np.zeros([N, N])

    # Update all columns, one at a time
    for var_index in range(N):

        # Build perturbation
        dx = np.zeros([N, 1])
        dx[var_index, 0] = h

        # Compute derivative
        der = (f(x + dx) - f(x)) / h

        # Update column
        J[:, var_index] = der[:, 0]

    return J


def find_root(
    f: callable, x0: np.ndarray, tol: float = 1e-9, max_iters: int = 20
) -> np.ndarray:
    """
    Find a root of f using the Newton-Raphson method.

    x0 must be a vector column (2D).
    """

    x = x0  # initial guess
    iters = 0  # iterations

    while np.linalg.norm(f(x), np.inf) > tol:

        x -= np.linalg.inv(jac(f, x)) @ f(x)  # Newton-Raphson step
        iters += 1

        if iters == max_iters:
            return 0.7 * x  # maybe this helps the optimizer

    return x


def power_mismatch(V3x: float, V3y: float, x: np.ndarray) -> np.ndarray:
    """
    Get power mismatch at bus 3.

    x must be a vector column (2D).
    """

    # Unpack known voltages
    V1x, V1y, V2x, V2y = x[:, 0]

    V1 = V1x + 1j * V1y
    V2 = V2x + 1j * V2y
    V3 = V3x + 1j * V3y

    # Get power coming from bus 1
    I13 = (V1 - V3) / (R1 + 1j * X1)
    S13 = V3 * np.conj(I13)

    # Repeat for bus 2
    I23 = (V2 - V3) / (R2 + 1j * X2)
    S23 = V3 * np.conj(I23)

    # Get active and reactive powers entering bus 3
    P_into_3 = np.real(S13 + S23)
    Q_into_3 = np.imag(S13 + S23)

    # Return mismatch vector
    return np.array([[P_into_3 - P3], [Q_into_3 - Q3]])


def get_V3(x: np.ndarray) -> tuple[float]:
    """
    Get real and imaginary parts of

    x must be a vector column (2D).
    """

    # Define initial estimate
    V30 = np.array([[1.0], [0.0]])

    # Define function whose root needs to be found
    def f(y: np.ndarray) -> np.ndarray:
        return power_mismatch(V3x=y[0, 0], V3y=y[1, 0], x=x)

    # Solve system of equations
    V3_sol = find_root(f=f, x0=V30)

    V3x = V3_sol[0, 0]
    V3y = V3_sol[1, 0]

    return V3x, V3y


def get_v_phasors(x: np.ndarray) -> float:
    """
    x must be a 1D array.
    """

    V1x, V1y, V2x, V2y = x

    # Get voltages
    V3x, V3y = get_V3(x=x.reshape((4, 1)))
    V1 = V1x + 1j * V1y
    V2 = V2x + 1j * V2y
    V3 = V3x + 1j * V3y

    return V1, V2, V3


def get_losses(x: np.ndarray) -> float:
    """
    x must be a 1D array.
    """

    V1, V2, V3 = get_v_phasors(x=x)

    I12 = (V1 - V3) / (R1 + 1j * X1)
    I23 = (V2 - V3) / (R2 + 1j * X2)

    return np.abs(I12) ** 2 * R1 + np.abs(I23) ** 2 * R2


def get_v_magnitudes(x: np.ndarray) -> tuple[float]:

    y = np.array([np.abs(V) for V in get_v_phasors(x=x)])

    return y


class Phasor:
    """
    A class for pretty-printing phasors.
    """

    def __init__(self, Vx: float, Vy: float, units: str) -> None:

        self.value = Vx + 1j * Vy
        self.units = units

    def __repr__(self) -> str:

        mag = np.abs(self.value)
        ang = np.rad2deg(np.angle(self.value))

        return f"{mag:.4f} {self.units} < {ang:.2f} degrees"

    def __str__(self) -> str:

        return self.__repr__()


if __name__ == "__main__":

    # Define parameters
    P3 = 5
    Q3 = 0
    R1 = 0.01
    X1 = 0.1
    R2 = R1
    X2 = X1

    # Initial guess, with x = np.array([V1x, V1y, V2x, V2y])
    x0 = np.array([1.0, 0.0, 1.0, 0.0])

    # Define constraints
    NLC = NonlinearConstraint(
        fun=get_v_magnitudes, lb=0.95 * np.ones(3), ub=1.05 * np.ones(3)
    )

    # Solve optimization
    res = minimize(fun=get_losses, x0=x0, constraints=[NLC])

    # Print results
    x_sol = res.x
    V3 = get_v_phasors(x=x_sol)[2]

    if res.success:
        print(f"Optimal V1 is {str(Phasor(x_sol[0], x_sol[1], units='pu'))}")
        print(f"Optimal V2 is {str(Phasor(x_sol[2], x_sol[3], units='pu'))}")
        print(f"Hence V3 = {str(Phasor(V3.real, V3.imag, units='pu'))}")
        print(f"Losses are {res.fun:.4f} pu")
    else:
        print(res.message)
