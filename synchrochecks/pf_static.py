"""
An object-oriented power-flow solver.

This program runs power-flow studies in networks of arbitrary size. It performs
slightly worse than well-established modules, such as pypower and pandapower,
but its syntax facilitates the creation of large-scale networks. Instead of
using indices to test for identity, look for neighbors, etc, the program
exploits object-oriented programming.

One salient feature of the solver is that it supports voltage-dependent loads.
The implementation uses the concept of an injector: an object connected to a PQ
bus whose active and reactive powers are arbitrary functions of the terminal
voltage. An injector can model devices such as smart inverters with P and Q
control.
"""


# Modules from the standard library
import copy
import bisect
import warnings
from collections.abc import Sequence

# Modules from this repository
import records
import utils

# Other modules
import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import tabulate

# numpy's settings
np.set_printoptions(linewidth=np.inf)


class StaticSystem:
    """
    A representation of the power system.

    This class is the main interface between the user and the power-flow
    solver. It contains methods for adding buses, lines, transformers,
    generators, injectors, and controllers. It also contains methods for
    running the power-flow study and importing data from ARTERE.
    """

    def __init__(
        self, name: str = "", pu: bool = True, base_MVA: float = 100
    ) -> None:
        """
        Initialize a system called 'name'.

        The argument 'pu' determines whether the system parameters are given in
        per-unit or in SI units.
        """

        # User-defined attributes
        self.name = name
        self.pu = pu
        self.base_MVA = base_MVA

        # Simulation-related attributes
        self.status = "unsolved"  # status of the power-flow study

        # Bus-related attributes
        self.slack = None
        self.PQ_buses = []
        self.PV_buses = []
        self.non_slack_buses = []  # equal to PQ_buses + PV_buses
        self.buses = []  # all buses = [slack] + non_slack_buses

        # Branch-related attributes
        self.lines = []
        self.transformers = []
        self.branches = []  # all branches = lines + transformers

        # Element-related attributes
        self.generators = []  # synchronous generators connected to PV buses
        self.injectors = []  # injectors connected to PQ buses

        # Dictionaries for quick access
        self.bus_dict = {}
        self.line_dict = {}
        self.transformer_dict = {}
        self.gen_dict = {}
        self.inj_dict = {}
        self.bus_to_injectors = {}
        self.bus_to_generators = {}

    def ohm2pu(self, Z_ohm: complex, base_kV: float) -> complex:
        """
        Convert impedance from ohms to pu (in the system's base).
        """

        base_impedance = base_kV**2 / self.base_MVA

        return Z_ohm / base_impedance

    def pu2ohm(self, Z_pu: complex, base_kV: float) -> complex:
        """
        Convert impedance from pu (in the system's base) to ohms.
        """

        base_impedance = base_kV**2 / self.base_MVA

        return Z_pu * base_impedance

    def mho2pu(self, Y_mho: complex, base_kV: float) -> complex:
        """
        Convert admittance 'Y' from mhos (siemens) to pu (in system's base).
        """

        return (
            0
            if Y_mho == 0
            else 1 / self.ohm2pu(Z_ohm=1 / Y_mho, base_kV=base_kV)
        )

    def pu2mho(self, Y_pu: complex, base_kV) -> complex:
        """
        Convert admittance 'Y' from pu (in system's base) to mhos (siemens).
        """

        return (
            0 if Y_pu == 0 else 1 / self.pu2ohm(Z_pu=1 / Y_pu, base_kV=base_kV)
        )

    def store_bus(self, bus: records.Bus) -> None:
        """
        Store bus keeping self.buses sorted: slack -> PQ -> PV.
        """

        if bus.name in self.bus_dict:
            raise RuntimeError(f"Bus {bus.name} already exists.")

        # Add bus to the list of buses (irrespective of bus type)
        bisect.insort(self.buses, bus)

        # Classify bus in remaining containers
        if isinstance(bus, records.Slack):
            self.slack = bus
        elif isinstance(bus, records.PQ):
            bisect.insort(self.PQ_buses, bus)
            bisect.insort(self.non_slack_buses, bus)
        elif isinstance(bus, records.PV):
            bisect.insort(self.PV_buses, bus)
            bisect.insort(self.non_slack_buses, bus)

        # Store in dictionary for quick access to named elements
        self.bus_dict[bus.name] = bus

        # Initialize entries in dictionaries for quick access to injectors and
        # generators
        self.bus_to_injectors[bus] = []
        self.bus_to_generators[bus] = []

    def remove_bus(self, bus: records.Bus) -> None:
        """
        Remove bus from the system.
        """

        # Remove bus from the list of buses (irrespective of bus type)
        self.buses.remove(bus)

        # Remove bus depending on its type
        if isinstance(bus, records.Slack):
            self.slack = None
        elif isinstance(bus, records.PQ):
            self.PQ_buses.remove(bus)
            self.non_slack_buses.remove(bus)
        elif isinstance(bus, records.PV):
            self.PV_buses.remove(bus)
            self.non_slack_buses.remove(bus)

        # Remove bus from dictionary for quick access to named elements
        del self.bus_dict[bus.name]

        # Remove injectors and generators connected to bus
        for inj in self.bus_to_injectors[bus]:
            self.remove_injector(inj)

        for gen in self.bus_to_generators[bus]:
            self.remove_generator(gen)

        # Remove entries in dictionaries for quick access to injectors and
        # generators
        del self.bus_to_injectors[bus]
        del self.bus_to_generators[bus]

    def replace_bus(self, old_bus: records.Bus, new_bus: records.Bus) -> None:
        """
        Replace bus in the system.
        """

        # Keep the only piece of information that is required from the old bus
        injectors = self.bus_to_injectors[old_bus]
        generators = self.bus_to_generators[old_bus]

        # Remove the old bus and add the new one
        self.remove_bus(bus=old_bus)
        self.store_bus(bus=new_bus)

        # Rewrite the injectors and generators associated to the old bus
        # (since store_bus erases them)
        self.bus_to_injectors[new_bus] = injectors
        self.bus_to_generators[new_bus] = generators

        # Replace the old bus in branches
        for branch in self.branches:
            if branch.from_bus is old_bus:
                branch.from_bus = new_bus
            if branch.to_bus is old_bus:
                branch.to_bus = new_bus

        # Replace the old bus in generators
        for gen in self.generators:
            if gen.bus is old_bus:
                gen.bus = new_bus

        # Replace the old bus in injectors
        for inj in self.injectors:
            if inj.bus is old_bus:
                inj.bus = new_bus


    def store_branch(self, branch: records.Branch) -> None:
        """
        Store branch keeping self.branches sorted: lines -> transformer.
        """

        if (
            branch.name in self.line_dict
            or branch.name in self.transformer_dict
        ):
            raise RuntimeError(f"Branch {branch.name} already exists.")

        # Add branch to the list of branches (irrespective of branch type)
        bisect.insort(self.branches, branch)

        # Classify branch in remaining containers. The addition to the
        # dictionaries is done for quick access to named elements.
        if branch.branch_type == "Line":
            self.lines.append(branch)
            self.line_dict[branch.name] = branch
        elif branch.branch_type == "Transformer":
            self.transformers.append(branch)
            self.transformer_dict[branch.name] = branch

    def remove_branch(self, branch: records.Branch) -> None:
        """
        Remove branch from the system.
        """

        # Remove branch from the list of branches (irrespective of branch type)
        self.branches.remove(branch)

        # Remove branch depending on its type
        if branch.branch_type == "Line":
            self.lines.remove(branch)
            del self.line_dict[branch.name]
        elif branch.branch_type == "Transformer":
            self.transformers.remove(branch)
            del self.transformer_dict[branch.name]

    def store_generator(self, gen: records.Generator) -> None:
        """
        Add a (large) generator to a PV or slack bus.

        It's very important that the generated power is in MW.

        There's no need to test for connection to the generator's bus because
        this is an argument of the generator's constructor anyway. Testing was
        important for injectors because they could be implemented by the user
        outside of this program.
        """

        if gen.name in self.gen_dict:
            raise RuntimeError(f"Generator {gen.name} already exists.")

        self.gen_dict[gen.name] = gen
        bisect.insort(self.generators, gen)
        bisect.insort(self.bus_to_generators[gen.bus], gen)

    def remove_generator(self, gen: records.Generator) -> None:
        """
        Remove generator from the system.
        """

        self.generators.remove(gen)
        del self.gen_dict[gen.name]
        self.bus_to_generators[gen.bus].remove(gen)

    def store_injector(self, inj: records.Injector) -> None:
        """
        Add an injector, which is anything that has the methods required below.

        It's very important that powers are in MVA and derivatives in MVA/pu.

        To facilitate exporting data, it's desirable that injectors have a
        prefix, a name, and a list of parameters.
        """

        if inj.name in self.inj_dict:
            raise RuntimeError(f"Injector {inj.name} already exists.")

        # Test if the injector is connected to a bus
        attr = getattr(inj, "bus", None)
        if not attr:
            raise RuntimeError(
                f"Each instance of {inj.__class__.__name__} "
                "must be connected to a bus"
            )

        # Test for missing methods
        for method in ["get_P", "get_Q", "get_dP_dV", "get_dQ_dV", "get_pars"]:
            attr = getattr(inj, method, None)
            if not attr or not callable(attr):
                raise RuntimeError(
                    f"Method {method} is missing from {inj.__class__.__name__}"
                )

        # Add injector
        self.inj_dict[inj.name] = inj
        bisect.insort(self.injectors, inj)
        bisect.insort(self.bus_to_injectors[inj.bus], inj)

    def remove_injector(self, inj: records.Injector) -> None:
        """
        Remove injector from the system.
        """

        self.injectors.remove(inj)
        del self.inj_dict[inj.name]
        self.bus_to_injectors[inj.bus].remove(inj)

    def add_slack(
        self,
        V_pu: float,
        name: str,
        theta_radians: float = 0,
        PL: float = 0,
        QL: float = 0,
        G: float = 0,
        B: float = 0,
        base_kV: float = np.nan,
        V_min_pu: float = 0.95,
        V_max_pu: float = 1.05,
        pu: bool = None,
    ) -> records.Bus:
        """
        Add the slack bus to the system.

        The boolean 'pu' determines whether PL, QL, G, and B are in per-unit or
        in SI units: MW, Mvar, mho, and mho, respectively.

        If 'pu' is not specified, the global setting is inherited.
        """

        if self.slack is not None:
            raise RuntimeError("The system already has a slack bus!")

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            G = self.mho2pu(Y_mho=G, base_kV=base_kV)
            B = self.mho2pu(Y_mho=B, base_kV=base_kV)
            PL = PL / self.base_MVA
            QL = QL / self.base_MVA

        # Initialize and store bus
        slack = records.Slack(
            V_pu=V_pu,
            theta_radians=theta_radians,
            PL_pu=PL,
            QL_pu=QL,
            G_pu=G,
            B_pu=B,
            base_kV=base_kV,
            bus_type="Slack",
            V_min_pu=V_min_pu,
            V_max_pu=V_max_pu,
            name=name,
        )

        self.store_bus(slack)

        # Returning the bus is important, so that the user can use the returned
        # value to specify the connectivity of branches and other elements.
        return slack

    def add_PQ(
        self,
        PL: float,
        QL: float,
        name: str,
        G: float = 0,
        B: float = 0,
        base_kV: float = np.nan,
        V_min_pu: float = 0.95,
        V_max_pu: float = 1.05,
        pu: bool = None,
    ) -> records.Bus:
        """
        Add a PQ (uncontrolled) bus to the system.

        The boolean 'pu' determines whether PL, QL, G, and B are in per-unit or
        in SI units: MW, Mvar, mho, and mho, respectively.

        If 'pu' is not specified, the global setting is inherited.
        """

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            G = self.mho2pu(Y_mho=G, base_kV=base_kV)
            B = self.mho2pu(Y_mho=B, base_kV=base_kV)
            PL = PL / self.base_MVA
            QL = QL / self.base_MVA

        # Initialize and store bus
        PQ = records.PQ(
            V_pu=1,  # voltage magnitude is computed later
            theta_radians=0,  # angle is computed later
            PL_pu=PL,
            QL_pu=QL,
            G_pu=G,
            B_pu=B,
            base_kV=base_kV,
            bus_type="PQ",
            V_min_pu=V_min_pu,
            V_max_pu=V_max_pu,
            name=name,
        )

        self.store_bus(PQ)

        # Returning the bus is important, so that the user can use the returned
        # value to specify the connectivity of branches and other elements.
        return PQ

    def add_PV(
        self,
        V_pu: float,
        PL: float,
        name: str,
        QL: float = 0,
        G: float = 0,
        B: float = 0,
        base_kV: float = np.nan,
        V_min_pu: float = 0.95,
        V_max_pu: float = 1.05,
        pu: bool = None,
    ) -> records.Bus:
        """
        Add a PV (controlled) bus to the system.

        The boolean 'pu' determines whether PL, QL, G, and B are in per-unit or
        in SI units: MW, Mvar, mho, and mho, respectively.

        If 'pu' is not specified, the global setting is inherited.
        """

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            G = self.mho2pu(Y_mho=G, base_kV=base_kV)
            B = self.mho2pu(Y_mho=B, base_kV=base_kV)
            PL = PL / self.base_MVA
            QL = QL / self.base_MVA

        # Build bus
        PV = records.PV(
            V_pu=V_pu,
            theta_radians=0,  # angle is computed later
            PL_pu=PL,
            QL_pu=QL,
            G_pu=G,
            B_pu=B,
            base_kV=base_kV,
            bus_type="PV",
            V_min_pu=V_min_pu,
            V_max_pu=V_max_pu,
            name=name,
        )

        self.store_bus(PV)

        return PV

    def add_line(
        self,
        from_bus: records.Bus,
        to_bus: records.Bus,
        X: float,
        name: str,
        R: float = 0,
        total_G: float = 0,
        total_B: float = 0,
        pu: bool = None,
        Snom_MVA: float = np.nan,
    ) -> records.Branch:
        """
        Add transmission line or transformer to the system.
        """

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            base_kV = from_bus.base_kV
            R = self.ohm2pu(Z_ohm=R, base_kV=base_kV)
            X = self.ohm2pu(Z_ohm=X, base_kV=base_kV)
            total_G = self.mho2pu(Y_mho=total_G, base_kV=base_kV)
            total_B = self.mho2pu(Y_mho=total_B, base_kV=base_kV)

        # Build a branch with n = 1
        total_Y = total_G + 1j * total_B
        branch = records.Branch(
            from_bus=from_bus,
            to_bus=to_bus,
            X_pu=X,
            R_pu=R,
            from_Y_pu=total_Y / 2,
            to_Y_pu=total_Y / 2,
            n_pu=1,
            branch_type="Line",
            Snom_MVA=Snom_MVA,
            name=name,
            sys=self,
        )

        # Add branch to the system
        self.store_branch(branch)

        return branch

    def add_transformer(
        self,
        from_bus: records.Bus,
        to_bus: records.Bus,
        X: float,
        name: str,
        R: float = 0,
        total_G: float = 0,
        total_B: float = 0,
        n_pu: float = 1,
        pu: bool = None,
        Snom_MVA: float = np.nan,
    ) -> records.Branch:
        """
        Add transformer to the system.

        The transformer is modeled as a branch with n != 1 (in general). The
        following convention is used:

        from   n:1   R+jX      to
        |------0 0---xxxx------|

        Ratio n is turns_from/turns_to. Impedance is on the side of the :1,
        i.e. the 'to' side. The 'to' voltage is thus used for normalization.
        """

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            base_kV = from_bus.base_kV
            R = self.ohm2pu(Z_ohm=R, base_kV=base_kV)
            X = self.ohm2pu(Z_ohm=X, base_kV=base_kV)
            total_G = self.mho2pu(Y_mho=total_G, base_kV=base_kV)
            total_B = self.mho2pu(Y_mho=total_B, base_kV=base_kV)

        # Build branch with (possibly) n != 1
        total_Y = total_G + 1j * total_B
        branch = records.Branch(
            from_bus=from_bus,
            to_bus=to_bus,
            X_pu=X,
            R_pu=R,
            from_Y_pu=total_Y / 2,
            to_Y_pu=total_Y / 2,
            n_pu=n_pu,
            branch_type="Transformer",
            Snom_MVA=Snom_MVA,
            name=name,
            sys=self,
        )

        # Add branch to the system
        self.store_branch(branch)

        return branch

    def build_Y(self) -> None:
        """
        Build the bus admittance matrix and store it as an attribute Y.

        Only elements that are in operation are taken into account. Note that
        transformers with off-nominal tap ratio lead to modified B_shunt,
        as explained in section 3.8 of Duncan Glover.

        Although Y is usually sparse, here it's stored as a numpy array because
        that's more readable. The conversion to a sparse-matrix datatype is
        done when computing the entries of the Jacobian.
        """

        # Initialize Y matrix
        N = len(self.buses)
        self.Y = np.zeros([N, N], dtype=complex)

        # Add contributions due to admittances at buses
        for i, bus in enumerate(self.buses):
            self.Y[i, i] += bus.G_pu + 1j * bus.B_pu

        # Add contributions from lines
        for line in self.lines:
            if line.in_operation:
                # Get bus indices
                i = self.buses.index(line.from_bus)
                j = self.buses.index(line.to_bus)
                # Get series impedance
                Y_series = 1 / (line.R_pu + 1j * line.X_pu)
                # Add contributions
                self.Y[i, i] += line.from_Y_pu + Y_series
                self.Y[j, j] += line.to_Y_pu + Y_series
                self.Y[i, j] -= Y_series
                self.Y[j, i] -= Y_series

        # Add contributions from transformers (requires taking n into account)
        for transformer in self.transformers:
            if transformer.in_operation:
                # Get bus indices
                i = self.buses.index(transformer.from_bus)
                j = self.buses.index(transformer.to_bus)
                # Get series impedance
                Y_series = 1 / (transformer.R_pu + 1j * transformer.X_pu)
                new_Y_series = Y_series / transformer.n_pu
                # Add contributions
                new_from_Y = (
                    transformer.from_Y_pu + Y_series
                ) / transformer.n_pu**2 - new_Y_series
                new_to_Y = transformer.to_Y_pu + Y_series - new_Y_series
                self.Y[i, i] += new_from_Y + new_Y_series
                self.Y[j, j] += new_to_Y + new_Y_series
                self.Y[i, j] -= new_Y_series
                self.Y[j, i] -= new_Y_series

    def build_dS_dV(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build partial derivatives for all buses.

        This method exploits sparsity and the fact that Y and V are available.

        For details, see https://matpower.org/docs/TN2-OPF-Derivatives.pdf
        """

        V = np.array([bus.get_phasor_V() for bus in self.buses])
        ib = range(len(V))
        Ybus = scipy.sparse.csr_matrix(self.Y)

        Ibus = Ybus * V
        diagV = scipy.sparse.csr_matrix((V, (ib, ib)))
        diagIbus = scipy.sparse.csr_matrix((Ibus, (ib, ib)))
        diagVnorm = scipy.sparse.csr_matrix((V / np.abs(V), (ib, ib)))

        dS_dVm = (
            diagV * np.conj(Ybus * diagVnorm) + np.conj(diagIbus) * diagVnorm
        )
        dS_dVa = 1j * diagV * np.conj(diagIbus - Ybus * diagV)

        # Convert back to arrays
        dS_dVm = dS_dVm.toarray()
        dS_dVa = dS_dVa.toarray()

        # Add new terms due to injectors
        for inj in self.injectors:
            i = self.buses.index(inj.bus)
            # Derivatives are substracted because S_injected is substracted
            # from delta_S
            dS_dVm[i, i] -= (
                inj.get_dP_dV() + 1j * inj.get_dQ_dV()
            ) / self.base_MVA

        return dS_dVm, dS_dVa

    def build_J(self) -> None:
        """
        Build Jacobian by calling dS_dV and extracting relevant derivatives.
        """

        dS_dVm, dS_dVa = self.build_dS_dV()

        M = len(self.PQ_buses)
        J11 = dS_dVa[1:, 1:].real
        J12 = dS_dVm[1:, 1 : M + 1].real
        J21 = dS_dVa[1 : M + 1, 1:].imag
        J22 = dS_dVm[1 : M + 1, 1 : M + 1].imag

        self.J = np.vstack([np.hstack([J11, J12]), np.hstack([J21, J22])])

    def build_full_J(self) -> None:
        """
        Build full Jacobian including slack and PV buses (usually not in J).

        One interesting fact is that

            self.full_J[:, :N].sum(axis=1) = zeros([N, 1])

        where N = len(self.buses). This happens because varying all angles by
        the same amount leaves the power flows unchanged.

        Since this matrix is not used in the actual power flow calculation, it
        is only meant to be called on demand, for instance when extracting
        sensitivities.
        """

        dS_dVm, dS_dVa = self.build_dS_dV()

        J11 = dS_dVa.real
        J12 = dS_dVm.real
        J21 = dS_dVa.imag
        J22 = dS_dVm.imag

        self.full_J = np.vstack([np.hstack([J11, J12]), np.hstack([J21, J22])])

    def get_S_towards_network(self) -> np.ndarray:
        """
        Return vector with powers exiting each bus towards the network.

        This method is useful for computing the power mismatches at every
        iteration of NR and for obtaining the consumption at each bus without
        having to distinguish between true and allocated load.
        """

        V = np.array([bus.get_phasor_V() for bus in self.buses])
        ib = range(len(V))
        Ybus = scipy.sparse.csr_matrix(self.Y)

        Ibus = Ybus * V
        diagV = scipy.sparse.csr_matrix((V, (ib, ib)))

        S_to_network_pu = diagV * np.conj(Ybus * np.asmatrix(V).T)

        return S_to_network_pu

    def build_F(self) -> None:
        """
        Build mismatch vector using the voltages that are currently available.

        The first M rows correspond to P and Q mismatches of PQ buses, whereas
        the remaining rows correspond to P of PV buses.
        """

        # Get total power 'injected' by loads (negative load)
        S_injected = np.array(
            [[-bus.PL_pu - 1j * bus.QL_pu] for bus in self.buses],
            dtype=complex,
        )

        # Add contributions due to injectors
        for inj in self.injectors:
            i = self.buses.index(inj.bus)
            S_injected[i, 0] += (
                inj.get_P() + 1j * inj.get_Q()
            ) / self.base_MVA

        # Add contributions due to generators
        for gen in self.generators:
            i = self.buses.index(gen.bus)
            S_injected[i, 0] += gen.PG_MW / self.base_MVA

        # Compute mismatch (ideally, power towards netw. - injected power = 0)
        delta_S = self.get_S_towards_network() - S_injected

        M = len(self.PQ_buses)
        F00 = delta_S[1:, 0].real
        F10 = delta_S[1 : M + 1, 0].imag

        self.F = np.vstack([F00, F10])

    def update_v(self, x: np.ndarray) -> None:
        """
        Update angle of all non-slack buses and voltage magnitude of PQ buses.
        """

        # Update angles
        for bus_no, bus in enumerate(self.non_slack_buses):
            bus.theta_radians = x[bus_no, 0]

        # Update magnitude
        for bus_no, bus in enumerate(self.PQ_buses):
            bus.V_pu = x[len(self.non_slack_buses) + bus_no, 0]

    def update_S(self) -> None:
        """
        Update P and Q consumption at all buses.

        To get the power demanded by an injector at a PQ bus, recall that it's
        possible to call get_P() and get_Q(), which will use the most recent
        voltage to get P and Q.
        """

        # Get power going to the network
        SL = -self.get_S_towards_network()

        # Add it as an attribute to the bus
        for bus_no, bus in enumerate(self.buses):
            bus.P_to_network_pu = -SL[bus_no, 0].real
            bus.Q_to_network_pu = -SL[bus_no, 0].imag

    def run_pf(
        self,
        tol: float = 1e-9,
        max_iters: int = 20,
        flat_start: bool = False,
        warn: bool = True,
    ) -> bool:
        """
        Run AC power-flow study using the Newton-Raphson method.

        The return value is True if the power flow converged and False
        otherwise.

        The threshold on the number of buses for applying LU factorization
        is hard-coded as 1000, based on experiments.
        """

        # Decide if LU decomposition is used when solving J @ dx = F
        LU_is_needed = len(self.buses) > 1000

        # Test for slack bus
        if self.slack is None:
            raise RuntimeError("The system must have a slack bus!")

        # Test for injectors at slack or PV buses
        for inj in self.injectors:
            if inj.bus not in self.PQ_buses:
                raise RuntimeError("Injectors can only be placed at PQ buses!")

        # Test for generators at slack or PQ buses. The condition
        # gen.PG_MW != 0 makes it possible to add dummy generators
        # (gen.PG_MW == 0) at the slack, which is useful, in turn, when
        # exporting into RAMSES.
        for gen in self.generators:
            if gen.bus not in self.PV_buses and gen.PG_MW != 0:
                raise RuntimeError(
                    "Generators can only be placed at PV buses!"
                )

        # Build nodal admittance matrix
        self.build_Y()

        # Ensure flat start
        x0 = np.vstack(
            [
                np.zeros([len(self.non_slack_buses), 1]),  # angles
                np.ones([len(self.PQ_buses), 1]),
            ]
        )  # magnitudes

        # Initialize
        x = x0
        if flat_start:
            self.update_v(x)
        iters = 0
        self.build_F()
        if np.linalg.norm(self.F, np.inf) < tol:  # test for lucky guess
            return True
        self.build_J()

        # Run Newton-Raphson method
        while np.linalg.norm(self.F, np.inf) > tol and iters < max_iters:
            # Update x
            if LU_is_needed:
                lu, piv = scipy.linalg.lu_factor(self.J)
                dx = scipy.linalg.lu_solve((lu, piv), self.F)
                x -= dx
            else:
                x -= np.matmul(np.linalg.inv(self.J), self.F)
            # Update operating point
            self.update_v(x)
            # Update matrices for next iteration
            self.build_F()
            self.build_J()
            # Count iteration
            iters += 1

        # Update complex powers
        self.update_S()

        # Update status
        if iters < max_iters:
            tol_W = round(tol * self.base_MVA * 1e6, 3)
            self.status = f"solved (max |F| < {tol_W} W) in {iters} iterations"
            return True
        else:
            self.status = f"non-convergent after {iters} iterations"
            if warn:
                warnings.warn(
                    f"Newton-Raphson did not converge after "
                    f"{iters} iterations."
                )
            return False

    @classmethod
    def import_ARTERE(
        cls,
        filename: str,
        system_name: str,
        base_MVA: float = 100,
        use_injectors: bool = False,
    ) -> "StaticSystem":
        """
        Import system from ARTERE file.

        The boolean 'use_injectors' determines whether loads and shunts are
        imported as injectors or as attributes of the buses.
        """

        # Initialize system and containers
        sys = cls(name=system_name, base_MVA=base_MVA)
        bus_types = {}  # bus_name: bus_type
        bus_objects = {}  # bus_name: bus_object
        v_setpoints = {}  # gen_name: V_pu
        generation_MW = {}  # gen_name: PG_MW
        locations = {}  # bus_name: location
        gen_names = {}  # bus_name: gen_name

        # Traverse file and save, initially, all buses as PQ buses
        with open(filename, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0] == "BUS":
                    bus_name = words[1]
                    bus_types[bus_name] = "PQ"
                # Take advantage of this iteration and store location of the
                # buses
                elif len(words) > 0 and words[0] == "BUSPART":
                    bus_name = words[2]
                    location = words[1]
                    locations[bus_name] = location

        # Traverse file again, correct for PV buses, and store generator data
        with open(filename, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0] == "GENER":
                    gen_name = words[1]
                    bus_name = words[2]
                    bus_types[gen_name] = "PV"
                    v_setpoints[gen_name] = float(words[6])
                    generation_MW[gen_name] = float(words[4])
                    gen_names[bus_name] = gen_name

        # Traverse file again and correct the slack
        with open(filename, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0] == "SLACK":
                    bus_name = words[1].strip(";")
                    bus_types[bus_name] = "Slack"

        # Traverse file again (last time) to call constructors and populate the
        # system
        with open(filename, "r") as f:
            for line in f:
                words = line.split()

                # Skip empty rows
                if len(words) == 0:
                    continue

                # Import buses
                if words[0] == "BUS":
                    # Read parameters
                    bus_name = words[1]
                    base_kV = float(words[2])
                    if use_injectors:
                        PL_MW = 0
                        QL_Mvar = 0
                        B_mho = 0
                    else:
                        PL_MW = float(words[3])
                        QL_Mvar = float(words[4])  # load Q
                        QS_Mvar = float(words[5])  # shunt Q
                        B_mho = utils.var2mho(Mvar_3P=QS_Mvar, kV_LL=base_kV)

                    # Call the right constructor
                    if bus_types[bus_name] == "PQ":
                        # Add bus
                        b = sys.add_PQ(
                            PL=PL_MW,
                            QL=QL_Mvar,
                            B=B_mho,
                            base_kV=base_kV,
                            name=bus_name,
                            pu=False,
                        )

                    elif bus_types[bus_name] == "PV":
                        # Add bus
                        b = sys.add_PV(
                            V_pu=v_setpoints[bus_name],
                            PL=PL_MW,
                            B=B_mho,
                            base_kV=base_kV,
                            name=bus_name,
                            pu=False,
                        )
                        # Add generator
                        gen = records.Generator(
                            PG_MW=generation_MW[bus_name],
                            bus=b,
                            name=gen_names[bus_name],
                        )
                        sys.store_generator(gen=gen)

                    elif bus_types[bus_name] == "Slack":
                        # Add bus
                        b = sys.add_slack(
                            V_pu=v_setpoints[bus_name],
                            B=B_mho,
                            base_kV=base_kV,
                            name=bus_name,
                            pu=False,
                        )
                        # Add dummy generator to the slack
                        gen = records.Generator(
                            PG_MW=0, bus=b, name=gen_names[bus_name]
                        )
                        sys.store_generator(gen)

                    # Save location and bus object
                    if bus_name in locations:
                        b.location = locations[bus_name]
                    bus_objects[bus_name] = b

                    # Create injectors
                    if use_injectors and bus_types[bus_name] == "PQ":
                        # Define load object
                        load_name = f"L{int(bus_name):02d}"
                        P_MW = float(words[3])
                        Q_Mvar = float(words[4])
                        if abs(P_MW) > 1e-6 or abs(Q_Mvar) > 1e-6:
                            load = records.Load(
                                name=load_name,
                                bus=b,
                                P0_MW=P_MW,
                                Q0_Mvar=Q_Mvar,
                            )
                            sys.store_injector(load)

                        # Define shunt object
                        QS_Mvar = float(words[5])
                        if abs(QS_Mvar) > 1e-9:
                            shunt_name = f"SH{int(bus_name):02d}"
                            shunt = records.Shunt(
                                name=shunt_name, bus=b, Mvar_at_Vnom=QS_Mvar
                            )
                            sys.store_injector(shunt)

                # Import lines
                elif words[0] == "LINE":
                    # Read parameters
                    line_name = words[1]
                    from_bus = bus_objects[words[2]]
                    to_bus = bus_objects[words[3]]
                    R_ohm = float(words[4])
                    X_ohm = float(words[5])
                    B_total_mho = 2 * float(words[6]) / 1e6
                    Snom_MVA = float(words[7])

                    # Call constructor
                    transmission_line = sys.add_line(
                        from_bus=from_bus,
                        to_bus=to_bus,
                        X=X_ohm,
                        R=R_ohm,
                        total_B=B_total_mho,
                        pu=False,
                        Snom_MVA=Snom_MVA,
                        name=line_name,
                    )

                # Import transformers
                elif words[0] == "TRFO":
                    trfo_name = words[1]
                    # Yes, these lines were flipped intentionally:
                    from_bus = bus_objects[words[3]]
                    to_bus = bus_objects[words[2]]
                    # Determine offset due to presence or absence of controlled
                    # bus (cumbersome, I know)
                    offset = 1 if "'" in words else 0
                    # Read remaining parameters
                    Snom_MVA = float(words[9 + offset])
                    R_perc = float(words[5 + offset])
                    X_perc = float(words[6 + offset])
                    R_pu = utils.change_base(
                        quantity=R_perc / 100,
                        base_MVA_old=Snom_MVA,
                        base_MVA_new=sys.base_MVA,
                        type="Z",
                    )
                    X_pu = utils.change_base(
                        quantity=X_perc / 100,
                        base_MVA_old=Snom_MVA,
                        base_MVA_new=sys.base_MVA,
                        type="Z",
                    )
                    # Transformers rarely have B:
                    n_pu = float(words[8 + offset]) / 100.0
                    # Call constructor
                    transformer = sys.add_transformer(
                        from_bus=from_bus,
                        to_bus=to_bus,
                        X=X_pu,
                        R=R_pu,
                        n_pu=n_pu,
                        pu=True,
                        Snom_MVA=Snom_MVA,
                        name=trfo_name,
                    )

                    # Read OLTC-related parameters
                    n_first_pu = float(words[10 + offset]) / 100
                    n_last_pu = float(words[11 + offset]) / 100
                    nb_pos = float(words[12 + offset])
                    half_db_pu = float(words[13 + offset])
                    v_setpoint_pu = float(words[14 + offset])

                    # If an OLTC is present, add it as an object
                    if n_first_pu * 100 > 0.5:
                        step_pu = (n_last_pu - n_first_pu) / (nb_pos - 1)
                        positions_up = round((n_last_pu - 1) / step_pu)
                        positions_down = round((1 - n_first_pu) / step_pu)
                        transformer.add_OLTC(
                            positions_up=positions_up,
                            positions_down=positions_down,
                            step_pu=step_pu,
                            v_setpoint_pu=v_setpoint_pu,
                            half_db_pu=half_db_pu,
                        )

        return sys

    def get_bus(self, name: str) -> records.Bus:
        """
        Get object associated to named bus.
        """

        if not isinstance(name, str):
            raise TypeError("Bus name must be a string.")

        if name not in self.bus_dict:
            raise RuntimeError(f"Bus {name} does not exist.")

        return self.bus_dict[name]

    def get_line(self, name: str) -> records.Branch:
        """
        Get object associated to named line.
        """

        if name not in self.line_dict:
            raise RuntimeError(f"Line {name} does not exist.")

        return self.line_dict[name]

    def get_transformer(self, name: str) -> records.Branch:
        """
        Get object associated to named transformer.
        """

        if name not in self.transformer_dict:
            raise RuntimeError(f"Transformer {name} does not exist.")

        return self.transformer_dict[name]

    def get_branches_between(
        self, bus_name_1: str, bus_name_2: str, warn: bool = True
    ) -> list[records.Branch]:
        """
        Return branches between two named buses.
        """

        bus_1 = self.get_bus(name=bus_name_1)
        bus_2 = self.get_bus(name=bus_name_2)

        candidates = [
            b
            for b in self.branches
            if (b.from_bus == bus_1 and b.to_bus == bus_2)
            or (b.from_bus == bus_2 and b.to_bus == bus_1)
        ]

        if warn and len(candidates) == 0:
            warnings.warn(
                f"There are no branches between {bus_name_1} and {bus_name_2}."
            )

        return candidates

    def get_generator(self, name: str) -> records.Generator:
        """
        Get object associated to named generator.
        """

        if name not in self.gen_dict:
            raise RuntimeError(f"Generator {name} does not exist.")

        return self.gen_dict[name]

    def get_injector(self, name: str) -> records.Injector:
        """
        Get object associated to named injector.
        """

        if name not in self.inj_dict:
            raise RuntimeError(f"Injector {name} does not exist.")

        return self.inj_dict[name]

    def get_bus_load_MVA(
        self, bus: records.Bus, attr: str = "P", tol: float = 1e-6
    ) -> float:
        """
        Return total (appreciable) bus load, with injectors as negative loads.
        """

        # Add loads from PL_pu
        total_load = self.base_MVA * getattr(bus, f"{attr}L_pu")

        # Possibly substract power injected by injectors
        total_load -= sum(
            getattr(inj, f"get_{attr}")()
            for inj in self.bus_to_injectors[bus]
            if isinstance(inj, records.Load)
        )

        return total_load if abs(total_load) > tol else None

    def get_bus_generation_MVA(
        self, bus: records.Bus, attr: str = "P", tol: float = 1e-4
    ) -> float:
        """
        Return total (appreciable) bus generation.
        """

        if isinstance(bus, records.Slack):
            return self.base_MVA * getattr(bus, f"{attr}_to_network_pu")

        elif isinstance(bus, records.PV):
            # If asking for P, read it from the generators
            if attr == "P":
                # Recall that generator P already is in MW
                total_gen = sum(
                    gen.PG_MW for gen in self.bus_to_generators[bus]
                )
                return total_gen if abs(total_gen) > tol else None
            # If asking for Q, use the one that flows towards the network
            elif attr == "Q":
                return self.base_MVA * bus.Q_to_network_pu

        elif isinstance(bus, records.PQ):
            return None

    def get_sensitive_load_MW_Mvar(
        self, bus: records.Bus
    ) -> tuple[float, float]:
        """
        Measure sensitive load (P, Q) at a particular bus.
        """

        sensitive_P_load_MW, sensitive_Q_load_Mvar = 0, 0

        for inj in self.bus_to_injectors[bus]:
            if isinstance(inj, records.Load):
                # Negative because get_P() and get_Q() return injected powers
                sensitive_P_load_MW -= inj.get_P()
                sensitive_Q_load_Mvar -= inj.get_Q()

        return sensitive_P_load_MW, sensitive_Q_load_Mvar

    def generate_table(
        self,
        show_buses: bool = True,
        show_lines: bool = True,
        show_transformers: bool = True,
        show_injectors: bool = True,
    ) -> str:
        """
        Display system data in tabular form.

        The net load can vary depending on the method chosen for simulating
        capacitors.

        If they were considered shunt admittances, they will not
        contribute to the net load, because it's as if they were part of the
        network, not a device that is connected to the bus.

        If, instead, they were considered as injectors, they will contribute
        to the net load.

        In any case, this does not affect the voltages. It's only a matter of
        displaying results.
        """

        # Fetch bus data
        if show_buses:
            bus_data = [
                [
                    self.buses.index(bus),
                    bus.name,
                    bus.bus_type,
                    bus.base_kV,
                    bus.V_pu,
                    np.rad2deg(bus.theta_radians),
                    self.get_bus_load_MVA(bus=bus, attr="P"),
                    self.get_bus_load_MVA(bus=bus, attr="Q"),
                    self.get_bus_generation_MVA(bus=bus, attr="P"),
                    self.get_bus_generation_MVA(bus=bus, attr="Q"),
                ]
                for bus in self.buses
            ]

            # Define headers
            bus_headers = [
                "\n\nIndex",
                "\n\nName",
                "\n\nType",
                "Nominal\nvoltage\n(kV)",
                "\nVoltage\n(pu)",
                "\nPhase\n(degrees)",
                "\nLoad\n(MW)",
                "\nLoad\n(Mvar)",
                "\nGeneration\n(MW)",
                "\nGeneration\n(Mvar)",
            ]

            # Build bus table
            bus_precision = (
                0,
                0,
                0,
                ".1f",
                ".4f",
                ".2f",
                ".3f",
                ".3f",
                ".3f",
                ".3f",
            )
            bus_table = tabulate.tabulate(
                tabular_data=bus_data,
                headers=bus_headers,
                floatfmt=bus_precision,
            )

        if show_lines:
            line_data = [
                [
                    self.lines.index(line),
                    line.name,
                    line.from_bus.name,
                    line.to_bus.name,
                    line.R_pu,
                    line.X_pu,
                    line.from_Y_pu.imag,
                    line.to_Y_pu.imag,
                    line.Snom_MVA,
                    line.get_pu_flows()[0] * self.base_MVA,
                    line.get_pu_flows()[1] * self.base_MVA,
                    line.get_pu_flows()[2] * self.base_MVA,
                    line.get_pu_flows()[3] * self.base_MVA,
                    line.get_pu_flows()[4] * self.base_MVA,
                ]
                for line in self.lines
            ]

            # Define headers
            line_headers = [
                "\nIndex",
                "\nName",
                "\nFrom bus",
                "\nTo bus",
                "\nR (pu)",
                "\nX (pu)",
                "B from\n(pu)",
                "B to\n(pu)",
                "Rating\n(MVA)",
                "P from\n(MW)",
                "Q from\n(Mvar)",
                "P to\n(MW)",
                "Q to\n(Mvar)",
                "Losses\n(MW)",
            ]

            # Build line table
            line_precision = (
                0,
                0,
                0,
                0,
                ".4f",
                ".4f",
                ".4f",
                ".4f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
            )
            line_table = tabulate.tabulate(
                tabular_data=line_data,
                headers=line_headers,
                floatfmt=line_precision,
            )

        if show_transformers:

            def has_OLTC(transformer: records.Branch) -> str:
                return "Yes" if transformer.has_OLTC else None

            transformer_data = [
                [
                    self.transformers.index(transformer),
                    transformer.name,
                    transformer.n_pu,
                    has_OLTC(transformer),
                    transformer.R_pu,
                    transformer.X_pu,
                    transformer.Snom_MVA,
                    transformer.get_pu_flows()[0] * self.base_MVA,
                    transformer.get_pu_flows()[1] * self.base_MVA,
                    transformer.get_pu_flows()[2] * self.base_MVA,
                    transformer.get_pu_flows()[3] * self.base_MVA,
                ]
                for transformer in self.transformers
            ]

            # Define headers
            transformer_headers = [
                "\nIndex",
                "\nName",
                "Ratio\n(pu)",
                "\nOLTC?",
                "\nR (pu)",
                "\nX (pu)",
                "Rating\n(MVA)",
                "P from\n(MW)",
                "Q from\n(Mvar)",
                "P to\n(MW)",
                "Q to\n(Mvar)",
            ]

            # Build transformer table
            transformer_precision = (
                0,
                0,
                ".2f",
                0,
                ".4f",
                ".4f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
            )
            transformer_table = tabulate.tabulate(
                tabular_data=transformer_data,
                headers=transformer_headers,
                floatfmt=transformer_precision,
            )

        if show_injectors:

            injector_data = [
                [
                    self.injectors.index(injector),
                    injector.name,
                    injector.bus.name,
                    injector.get_P() if not np.isclose(injector.get_P(), 0) else 0.0,
                    injector.get_Q() if not np.isclose(injector.get_Q(), 0) else 0.0,
                    injector.get_dP_dV() if not np.isclose(injector.get_dP_dV(), 0) else 0.0,
                    injector.get_dQ_dV() if not np.isclose(injector.get_dQ_dV(), 0) else 0.0
                ]
                for injector in self.injectors
            ]

            # Define headers
            injector_headers = [
                "Index",
                "Name",
                "Bus",
                "P (MW)",
                "Q (Mvar)",
                "dP/dV (MW/pu)",
                "dQ/dV (Mvar/pu)",
            ]

            # Build injector table
            injector_precision = (
                0,
                0,
                0,
                ".3f",
                ".3f",
                ".3f",
                ".3f",
            )
            injector_table = tabulate.tabulate(
                tabular_data=injector_data,
                headers=injector_headers,
                floatfmt=injector_precision,
            )

        # Possibly add a filler name for the system
        if self.name == "":
            display_name = str(len(self.buses)) + "-bus system"
        else:
            display_name = self.name

        # Report the status (including convergence and number of iterations)
        display_status = "Status: " + self.status

        # Build output string
        output_str = (
            f"\n{display_name}\n\n{display_status}\n\n"
            f"BUS DATA:\n\n{bus_table}\n"
        )

        if show_lines:
            output_str += f"\n\nLINE DATA:\n\n\n{line_table}\n"

        if show_transformers:
            output_str += f"\n\nTRANSFORMER DATA:\n\n\n{transformer_table}\n"

        if show_injectors:
            output_str += f"\n\nINJECTOR DATA:\n\n\n{injector_table}\n"

        return output_str

    def __str__(self) -> str:
        """
        Print only the bus data.
        """

        return self.generate_table(
            # show_buses=True, show_lines=False, show_transformers=False
        )
