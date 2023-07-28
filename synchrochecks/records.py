"""
Classes for modelling equipment and exporting it seamlessly to RAMSES.
"""

from utils import change_base  # module from src/utils.py
from re import sub  # module from stdli
from textwrap import fill  # module from stdli
import numpy as np

# The first two classes are mostly used for pretty-printing in RAMSES. The
# remaining classes are used for modeling equipment.


class Parameter:
    """
    Class for all numeric parameters used by records.

    One advantage of having them being objects instead of simple floats is that
    the user is forced to define their name (used when pretty-printing a
    header) and the significant digits (which leads to cleaner RAMSES files).
    """

    def __init__(self, name: str, value: float, digits: int = 6) -> None:
        """
        Define a parameter called 'name' with a number of significant digits.
        """

        self.name = name
        self.value = value
        self.digits = digits

    def __str__(self) -> str:
        """
        Cast parameter to string with enough significant digits (if numeric).
        """

        # If the parameter is a string (e.g. for synchronous machines), return
        # it as is
        if isinstance(self.value, str):
            return self.value

        # The following is deprecated. For simplicity, digits now takes the
        # default value of 6.
        # If no significant digits are specified, return the parameter as is
        elif self.digits is None:
            return str(float(self.value))

        # Otherwise, return the parameter with the specified number of digits
        else:
            # A zero is always printed as 0.0
            if self.value == 0:
                return str(0.0)

            # Parse sign, number, and power (related to the number of zeros)
            sign = "-" if self.value < 0 else ""
            num = abs(float(self.value))
            power = np.log10(num)

            # Place the decimal separator in the right place
            if num < 1:
                # Decimal separator is to the left of the number
                step = int(10 ** (-int(power) + self.digits) * num)
                return (
                    sign
                    + "0."
                    + "0" * -int(power)
                    + str(int(step)).rstrip("0")
                )

            # Alternative: some significant digits are present after the
            # decimal separator
            elif power < self.digits - 1 and not isinstance(self.value, int):
                # Decimal separator is in the middle
                result = sign + ("{0:." + str(self.digits) + "g}").format(num)
                if "." not in result:
                    result += ".0"
                return result

            # Alternative: no significant digit is present after the decimal
            # separator
            else:
                # Decimal separator is not present at all
                return sign + str(int(num)) + ".0"


class Record:
    """
    Parent class for user-defined records.

    The only thing that all records have in common is their printing method,
    which is constrained by RAMSES' syntax. All other attributes are defined
    in the child classes.
    """

    def __lt__(self, other: "Record") -> bool:
        return f"{self.prefix} {self.name}" < f"{other.prefix} {other.name}"

    def __str__(self) -> str:
        fields = [self.prefix, self.name] + [str(p) for p in self.get_pars()]

        # Add default delimiter if not specified
        if hasattr(self, "delimiter"):
            delim = self.delimiter
        else:
            delim = ";"

        # Add default indentation if not specified
        if hasattr(self, "ind_offset"):
            offset = self.ind_offset
        else:
            offset = ""

        text = " ".join(fields) + delim

        # Remove duplicate spaces. This is done using a regular expression that
        # I found on StackOverflow.
        text = sub(" +", " ", text).replace(" ;", ";")

        # Wrap record information to 80 characters with an offset of 0 spaces
        # for the first line and 4 spaces for the remaining lines
        return fill(
            text,
            initial_indent=offset,
            subsequent_indent=offset + "    ",
            width=80,
        )


class Injector(Record):
    """
    Parent class for user-defined injectors.

    This class simply inherits __str__() from the Record class.

    All injectors will inherit null values for P, Q, dP_dV, and dQ_dV, unless
    overwritten.
    """

    prefix: str = ""

    def get_P(self) -> float:
        """
        Return active power injection in MW.
        """

        return 0

    def get_Q(self) -> float:
        """
        Return reactive power injection in Mvar.
        """

        return 0

    def get_dP_dV(self) -> float:
        """
        Return derivative of active power injection w.r.t. voltage magnitude.

        Units are MW/pu.
        """

        return 0

    def get_dQ_dV(self) -> float:
        """
        Return derivative of reactive power injection w.r.t. voltage magnitude.

        Units are Mvar/pu.
        """

        return 0


class DCTL(Record):
    """
    Parent class for DCTLs (discrete controllers) in RAMSES.

    It simply inherits __str__() from the Record class.

    This class is useful in experiment.py, to recognize the observable type.
    """

    prefix: str = "DCTL"

    def get_pars(self) -> list[Parameter]:
        return []


class Bus(Record):
    """
    A class for buses of any kind: Slack, PV, or PQ.
    """

    prefix: str = "BUS"

    def __init__(
        self,
        V_pu: float,
        theta_radians: float,
        PL_pu: float,
        QL_pu: float,
        G_pu: float,
        B_pu: float,
        base_kV: float,
        bus_type: str,
        V_min_pu: float,
        V_max_pu: float,
        name: str,
    ) -> None:
        """
        Options for bus_type are 'Slack', 'PV', and 'PQ'.
        """

        # Set all keyword arguments as attributes
        attributes = vars()
        for key in attributes:
            setattr(self, key, attributes[key])

        # Set attributes used internally
        self.P_to_network_pu = np.nan
        self.Q_to_network_pu = np.nan
        self.allocated_PL_pu = PL_pu
        self.allocated_QL_pu = QL_pu
        self.scheduled_v_pu = (
            V_pu if bus_type == "Slack" or bus_type == "PV" else ""
        )
        self.location = ""
        self.pars = [Parameter("nominal_kV", base_kV)]
        self.has_stiff_load = False
        self.is_connected = True  # toggle to False if bus ends up in an island

    def get_phasor_V(self) -> complex:
        return self.V_pu * np.exp(1j * self.theta_radians)

    def get_pars(self) -> list[Parameter]:
        return [Parameter("nominal_kV", self.base_kV)]

    def change_base(self,
                    base_MVA_old: float,
                    base_MVA_new: float,
                    base_kV_old: float,
                    base_kV_new: float) -> None:
        """
        Change the power of the bus.
        """

        # Change base voltage
        self.base_kV = base_kV_new

        # Change powers in pu
        for attr in ["PL_pu", "QL_pu", "allocated_PL_pu", "allocated_QL_pu"]:
            setattr(
                self,
                attr,
                change_base(
                    quantity=getattr(self, attr),
                    base_MVA_old=base_MVA_old,
                    base_MVA_new=base_MVA_new,
                    base_kV_old=base_kV_old,
                    base_kV_new=base_kV_new,
                    type="S",
                ),
            )

        # Change admittances in pu
        for attr in ["G_pu", "B_pu"]:
            setattr(
                self,
                attr,
                change_base(
                    quantity=getattr(self, attr),
                    base_MVA_old=base_MVA_old,
                    base_MVA_new=base_MVA_new,
                    base_kV_old=base_kV_old,
                    base_kV_new=base_kV_new,
                    type="Y",
                ),
            )


class Slack(Bus):
    """
    Slack bus, which is a bus with a fixed voltage magnitude and angle.

    In the sorted list of buses, the slack bus is always the first one,
    hence the implementation of __lt__().
    """

    def __lt__(self, other: Bus) -> bool:
        """
        Slack bus must always come first in the sorted list of buses.
        """

        return True


class PQ(Bus):
    """
    PQ bus, which has fixed active and reactive power injections.
    """

    def __lt__(self, other: Bus) -> bool:
        """
        PQ buses must always come between the slack bus and the PV buses.

        Another way of saying this is that the PQ buses must always come
        after the existing slack and PQ buses, but before PV buses.
        """

        return isinstance(other, PV)


class PV(Bus):
    """
    PV bus, which has fixed active power injection and voltage magnitude.
    """

    def __lt__(self, other: Bus) -> bool:
        """
        PV buses must always come last in the sorted list of buses.
        """

        return False


class Thevenin(Record):
    """
    Thevening equivalent seen from a bus.
    """

    prefix: str = "INJEC THEVEQ"

    def __init__(self, name: str, bus: Bus) -> None:
        self.name = name
        self.bus = bus

    def get_pars(self) -> list[Parameter]:
        return [
            Parameter("bus", self.bus.name),
            Parameter("FP", 1),
            Parameter("FQ", 1),
            Parameter("P", 0),
            Parameter("Q", 0),
            Parameter("MVA", 10_000_000),
        ]


class Frequency(Record):
    """
    Nominal frequency of the system.

    Like any other class that inherits from Record, it is only used when
    exporting to RAMSES.
    """

    prefix: str = "FNOM"

    def __init__(self, fnom: float) -> None:
        self.name = ""
        self.pars = [Parameter("FNOM", fnom)]

    def get_pars(self) -> list[Parameter]:
        return self.pars


class InitialVoltage(Record):
    """
    Initial voltage of a bus, obtained from a power flow.
    """

    prefix: str = "LFRESV"

    def __init__(self, bus: Bus) -> None:
        self.name = bus.name
        self.pars = [
            Parameter("VMAG_pu", bus.V_pu, digits=8),
            Parameter("VANG_rad", bus.theta_radians, digits=8),
        ]

    def get_pars(self) -> list[Parameter]:
        return self.pars


class SYNC_MACH(Record):
    """
    Synchronous machines, specifically the SYNC_MACH record in RAMSES.
    """

    prefix: str = "SYNC_MACH"

    def __init__(
        self,
        name: str,
        bus: Bus,
        Snom_MVA: float,
        Pnom_MW: float,
        H: float,
        D: float,
        IBRATIO: float,
        model: str,
        Xl: float,
        Xd: float,
        Xdp: float,
        Xdpp: float,
        Xq: float,
        Xqp: float,
        Xqpp: float,
        m: float,
        n: float,
        Ra: float,
        Td0p: float,
        Td0pp: float,
        Tq0p: float,
        Tq0pp: float,
    ) -> None:
        attributes = vars()
        for key in attributes:
            setattr(self, key, attributes[key])

        self.name = name  # possibly redundant
        self.bus = bus  # possibly redundant
        self.delimiter = ""  # empty since the exciter comes next

    def get_pars(self) -> list[Parameter]:
        pars = [
            Parameter("bus_name", self.bus.name),
            Parameter("FP", 1),
            Parameter("FQ", 1),
            Parameter("P", 0),
            Parameter("Q", 0),
        ]

        forbidden = ["self", "prefix", "name", "bus", "delimiter"]

        for attr in self.__dict__:
            if attr not in forbidden:
                par = Parameter(attr, self.__dict__[attr])
                pars.append(par)

        return pars


class EXC:
    """
    Parent class for exciters.
    """

    pass


class GENERIC1(EXC, Record):
    """
    Exciter model GENERIC1 in RAMSES.
    """

    prefix: str = "EXC GENERIC1"

    def __init__(
        self,
        iflim: float,
        d: float,
        f: float,
        s: float,
        k1: float,
        k2: float,
        L1: float,
        L2: float,
        G: float,
        Ta: float,
        Tb: float,
        Te: float,
        L3: float,
        L4: float,
        SPEEDIN: float,
        KPSS: float,
        Tw: float,
        T1: float,
        T2: float,
        T3: float,
        T4: float,
        DVMIN: float,
        DVMAX: float,
    ) -> None:
        attributes = vars()
        for key in attributes:
            setattr(self, key, attributes[key])

        self.name = ""
        self.delimiter = ""  # empty since the governor comes next
        self.ind_offset = "    "

    def get_pars(self) -> list[Parameter]:
        pars = []

        forbidden = ["self", "prefix", "name", "ind_offset", "delimiter"]

        for attr in self.__dict__:
            if attr not in forbidden:
                par = Parameter(attr, self.__dict__[attr])
                pars.append(par)

        return pars


class TOR:
    """
    Parent class for governors.
    """

    pass


class CONSTANT(TOR, Record):
    """
    Governor model CONSTANT in RAMSES.
    """

    prefix: str = "TOR CONSTANT"

    def __init__(self) -> None:
        self.name = ""
        self.delimiter = ";"  # no record comes next
        self.ind_offset = "    "

    def get_pars(self) -> list[Parameter]:
        """
        A constant governor has no parameters.
        """

        return []


class HYDRO_GENERIC1(TOR, Record):
    """
    Governor model HYDRO_GENERIC1 in RAMSES.
    """

    prefix: str = "TOR HYDRO_GENERIC1"

    def __init__(
        self,
        sigma: float,
        Tp: float,
        Qv: float,
        Kp: float,
        Ki: float,
        Tsm: float,
        limzdot: float,
        Tw: float,
    ) -> None:
        attributes = vars()
        for key in attributes:
            setattr(self, key, attributes[key])

        self.name = ""
        self.delimiter = ";"  # no record comes next
        self.ind_offset = "    "

    def get_pars(self) -> list[Parameter]:
        pars = []

        forbidden = ["self", "prefix", "name", "ind_offset", "delimiter"]

        for attr in self.__dict__:
            if attr not in forbidden:
                par = Parameter(attr, self.__dict__[attr])
                pars.append(par)

        return pars


class Generator(Record):
    """
    Generator at PV bus, linked to a machine, an exciter, and a governor.
    """

    prefix: str = "SYNC_MACH"

    def __init__(self, PG_MW: float, bus: Bus, name: str) -> None:
        if not isinstance(bus, PV) and not isinstance(bus, Slack):
            raise ValueError(
                "Generator must be connected to a generation bus."
            )

        self.bus = bus
        self.PG_MW = PG_MW
        self.name = name
        self.pars = []
        self.machine = None
        self.exciter = None
        self.governor = None
        self.in_operation = True
        self.location = ""

    def __str__(self) -> str:
        """
        This method is reimplemented because printing generators is hard.

        Much of the difficulties were moved to the classes of machines,
        exciters and governors.
        """

        return f"{self.machine}\n{self.exciter}\n{self.governor}"

    def trip(self) -> None:
        """
        Trip (disconnect) generator.
        """

        self.in_operation = False

    def trip_back(self) -> None:
        """
        Trip generator back (reconnect).
        """

        self.in_operation = True


class Shunt(Injector):
    """
    Shunt compensator connected to a bus.

    Although numerically inefficient, modeling them as objects is useful
    for keeping track of them.

    See the Injector class for the units of get_P(), get_Q(), get_dP_dV(), and
    get_dQ_dV().
    """

    prefix: str = "SHUNT"

    def __init__(self, name: str, bus: Bus, Mvar_at_Vnom: float) -> None:
        self.name = name
        self.bus = bus
        self.Mvar_at_Vnom = Mvar_at_Vnom
        self.in_operation = True

    def trip(self) -> None:
        self.in_operation = False

    def trip_back(self) -> None:
        self.in_operation = True

    def get_Q(self) -> float:
        return (self.bus.V_pu) ** 2 * self.Mvar_at_Vnom

    def get_dQ_dV(self) -> float:
        """
        Numerical derivative of get_Q(), obtained analytically.
        """

        return 2 * self.bus.V_pu * self.Mvar_at_Vnom

    def get_pars(self) -> list[Parameter]:
        return [
            Parameter("BUS_NAME", self.bus.name),
            Parameter("Mvar_at_Vnom", self.Mvar_at_Vnom),
            Parameter("breaker_status", 1 if self.in_operation else 0),
        ]


class Load(Injector):
    """
    Voltage-dependent load connected to a bus.

    For the sake of simplicity, loads are always initialized as constant, but
    this can be changed by overwriting V0, alpha, and beta.

    See the Injector class for the units of get_P(), get_Q(), get_dP_dV(), and
    get_dQ_dV().
    """

    prefix: str = "INJEC LOAD"

    def __init__(
        self, name: str, bus: Bus, P0_MW: float, Q0_Mvar: float
    ) -> None:
        self.allocated_P0_MW = P0_MW
        self.allocated_Q0_Mvar = Q0_Mvar
        self.V0 = 1
        self.alpha = 0
        self.beta = 0
        self.name = name
        self.bus = bus
        self.P0_MW = P0_MW
        self.Q0_Mvar = Q0_Mvar
        self.DP = 0
        self.A1 = 1
        self.alpha1 = self.alpha
        self.A2 = 0
        self.alpha2 = 0
        self.alpha3 = 0
        self.DQ = 0
        self.B1 = 1
        self.beta1 = self.beta
        self.B2 = 0
        self.beta2 = 0
        self.beta3 = 0

    def make_voltage_sensitive(self, alpha: float, beta: float) -> None:
        """
        Turn load into voltage-sensitive load using latest power flow solution.
        """

        self.alpha = self.alpha1 = alpha
        self.beta = self.beta1 = beta
        self.V0 = self.bus.V_pu

    def increment_P_by(self, delta_P_MW: float) -> None:
        """
        Increment active power by a value.
        """

        self.P0_MW += delta_P_MW
        self.allocated_P0_MW += delta_P_MW

    def increment_Q_by(self, delta_Q_Mvar: float) -> None:
        """
        Increment reactive power by a value.
        """

        self.Q0_Mvar += delta_Q_Mvar
        self.allocated_Q0_Mvar += delta_Q_Mvar

    def set_P_to(self, PL_MW: float) -> None:
        """
        Set active power to a value.
        """

        self.P0_MW = PL_MW
        self.allocated_P0_MW = PL_MW

    def set_Q_to(self, QL_Mvar: float) -> None:
        """
        Set reactive power to a value.
        """

        self.Q0_Mvar = QL_Mvar
        self.allocated_Q0_Mvar = QL_Mvar

    def scale_P_by(self, factor: float) -> None:
        """
        Scale active load by a factor.
        """

        self.P0_MW *= factor
        self.allocated_P0_MW *= factor

    def scale_Q_by(self, factor: float) -> None:
        """
        Scale reactive load by a factor.
        """

        self.Q0_Mvar *= factor
        self.allocated_Q0_Mvar *= factor

    def get_P(self) -> float:
        """
        Exponential load model.
        """

        return -self.P0_MW * (self.bus.V_pu / self.V0) ** self.alpha

    def get_Q(self) -> float:
        """
        Exponential load model.
        """

        return -self.Q0_Mvar * (self.bus.V_pu / self.V0) ** self.beta

    def get_dP_dV(self) -> float:
        """
        Numerical derivative of get_P(), obtained analytically.
        """

        return (
            -self.P0_MW
            * self.alpha
            * (self.bus.V_pu / self.V0) ** (self.alpha - 1)
            / self.V0
        )

    def get_dQ_dV(self) -> float:
        """
        Numerical derivative of get_Q(), obtained analytically.
        """

        return (
            -self.Q0_Mvar
            * self.beta
            * (self.bus.V_pu / self.V0) ** (self.beta - 1)
            / self.V0
        )

    def get_pars(self) -> list[Parameter]:
        """
        The load power is specified directly, as with other injectors.
        """

        return [
            Parameter("bus", self.bus.name),
            Parameter("FP", 0),
            Parameter("FQ", 0),
            Parameter("P", self.get_P(), digits=10),
            Parameter("Q", self.get_Q(), digits=10),
            Parameter("DP", self.DP),
            Parameter("A1", self.A1),
            Parameter("alpha1", self.alpha1),
            Parameter("A2", self.A2),
            Parameter("alpha2", self.alpha2),
            Parameter("alpha3", self.alpha3),
            Parameter("DQ", self.DQ),
            Parameter("B1", self.B1),
            Parameter("beta1", self.beta1),
            Parameter("B2", self.B2),
            Parameter("beta2", self.beta2),
            Parameter("beta3", self.beta3),
        ]


class Branch(Record):
    """
    Branch of any kind: line or transformer.

    A branch becomes a transformer as soon as n != 1. This can happen either
    when calling the constructor or after adding an OLTC.

    By default, all branches are initialized as being 'in operation'; if this
    attribute is changed to False, then the branch is no longer taken into
    account when building the Y matrix. This attribute is also used when
    building the system's multigraph, used for checking connectivity.
    """

    def __init__(
        self,
        from_bus: Bus,
        to_bus: Bus,
        X_pu: float,
        R_pu: float,
        from_Y_pu: complex,
        to_Y_pu: complex,
        n_pu: float,
        branch_type: str,
        Snom_MVA: float,
        name: str,
        sys: "StaticSystem",
    ) -> None:
        """
        branch_type can be Line or Transformer; sys is pf_static.StaticSystem.
        """

        # Set all keyword arguments as attributes
        attributes = vars()
        for key in attributes:
            setattr(self, key, attributes[key])

        # Use branch type to set prefix
        if branch_type == "Line":
            self.prefix = "LINE"
        elif branch_type == "Transformer":
            self.prefix = "TRFO"
        else:
            raise ValueError("Branch type must be 'Line' or 'Transformer'.")

        self.in_operation = True
        self.has_OLTC = False

    def __lt__(self, other: "Branch") -> bool:
        """
        Sort branches so that lines come before transformers.
        """

        return other.branch_type == "Transformer"

    def replace_bus_by(self, old_bus: Bus, new_bus: Bus) -> None:
        """
        Repalce old_bus by new_bus.
        """

        if old_bus is self.from_bus:
            self.from_bus = new_bus
        elif old_bus is self.to_bus:
            self.to_bus = new_bus

    def change_base(self,
                    base_MVA_old: float,
                    base_MVA_new: float,
                    base_kV_old: float,
                    base_kV_new: float) -> None:
        """
        Change the power base of the branch.
        """

        if self.branch_type == "Transformer" and self.n_pu != 1:
            raise NotImplementedError(
                "Changing the base of a transformer is not implemented."
            )

        # Change impedances
        for attr in ["R_pu", "X_pu"]:
            setattr(
                self,
                attr,
                change_base(
                    quantity=getattr(self, attr),
                    base_MVA_old=base_MVA_old,
                    base_MVA_new=base_MVA_new,
                    base_kV_old=base_kV_old,
                    base_kV_new=base_kV_new,
                    type="Z",
                ),
            )

        # Change admittances
        for attr in ["from_Y_pu", "to_Y_pu"]:
            setattr(
                self,
                attr,
                change_base(
                    quantity=getattr(self, attr),
                    base_MVA_old=base_MVA_old,
                    base_MVA_new=base_MVA_new,
                    base_kV_old=base_kV_old,
                    base_kV_new=base_kV_new,
                    type="Y",
                ),
            )

    def touches(self, location: str) -> bool:
        """
        Check if a branch touches a certain location (RAMSES region).
        """

        return location in {self.from_bus.location, self.to_bus.location}

    def disconnect(self) -> None:
        """
        Disconnect branch and update connectivity of all buses.
        """

        self.in_operation = False
        self.sys.update_connectivity(reference_bus=self.sys.slack)

    def connect(self) -> None:
        """
        Connect branch again and update connectivity of all buses.
        """

        self.in_operation = True
        self.sys.update_connectivity(reference_bus=self.sys.slack)

    def get_HV_bus(self) -> Bus:
        """
        Return the bus with the highest base voltage.
        """

        return (
            self.from_bus
            if self.from_bus.base_kV > self.to_bus.base_kV
            else self.to_bus
        )

    def get_LV_bus(self) -> Bus:
        """
        Return the bus with the lowest base voltage.
        """

        return (
            self.from_bus
            if self.from_bus.base_kV < self.to_bus.base_kV
            else self.to_bus
        )

    def get_pu_flows(self) -> tuple[float, float, float, float, float]:
        """
        Get (P_from, Q_from, P_to, Q_to) in MW and Mvar, entering the branch.
        """

        V_from = self.from_bus.get_phasor_V()
        V_to = self.to_bus.get_phasor_V()

        if self.branch_type == "Line":
            # If the branch is a line, the parameters of the pi model can be
            # read directly.
            Y_pu_series = 1 / (self.R_pu + 1j * self.X_pu)
            from_Y_pu = self.from_Y_pu
            to_Y_pu = self.to_Y_pu

        elif self.branch_type == "Transformer":
            # On the other hand, if the branch is a transformer, the pi model
            # must be built first, taking into account the turns ratio.
            Y_pu_series = 1 / (self.R_pu + 1j * self.X_pu)
            new_Y_pu_series = Y_pu_series / self.n_pu
            new_from_Y_pu = (
                self.from_Y_pu + Y_pu_series
            ) / self.n_pu**2 - new_Y_pu_series
            new_to_Y_pu = self.to_Y_pu + Y_pu_series - new_Y_pu_series
            # We now store these values into the original variables (the same)
            # names used in the "if" branch.
            Y_pu_series = new_Y_pu_series
            from_Y_pu = new_from_Y_pu
            to_Y_pu = new_to_Y_pu

        # The remaining calculations are the same for both branches and simply
        # apply circuit theory.

        I_from = (V_from - V_to) * Y_pu_series + V_from * from_Y_pu
        I_to = (V_to - V_from) * Y_pu_series + V_to * to_Y_pu

        S_from = V_from * np.conj(I_from)
        S_to = V_to * np.conj(I_to)

        P_from = S_from.real
        Q_from = S_from.imag
        P_to = S_to.real
        Q_to = S_to.imag

        P_losses = max(abs(P_from), abs(P_to)) - min(abs(P_from), abs(P_to))

        return P_from, Q_from, P_to, Q_to, P_losses

    def add_OLTC(
        self,
        positions_up: int,
        positions_down: int,
        step_pu: float,
        v_setpoint_pu: float = 1,
        half_db_pu: float = 0.01,
    ) -> None:
        """
        Add an on-load tap changer (OLTC) to the (transformer) branch.
        """

        self.OLTC = OLTC(
            trafo=self,
            positions_up=positions_up,
            positions_down=positions_down,
            step_pu=step_pu,
            v_setpoint_pu=v_setpoint_pu,
            half_db_pu=half_db_pu,
        )
        self.has_OLTC = True

    def get_pars(self) -> list[Parameter]:
        if self.branch_type == "Line":
            base_kV = self.from_bus.base_kV
            half_WC_pu = np.imag(self.from_Y_pu)

            return [
                Parameter("bus1", self.from_bus.name),
                Parameter("bus2", self.to_bus.name),
                Parameter("R", self.sys.pu2ohm(self.R_pu, base_kV), digits=6),
                Parameter("X", self.sys.pu2ohm(self.X_pu, base_kV), digits=6),
                Parameter(
                    "WC/2",
                    1e6 * self.sys.pu2mho(half_WC_pu, base_kV),
                    digits=6,
                ),
                Parameter("Snom_MVA", self.Snom_MVA),
                Parameter("br1", 1 if self.in_operation else 0),
                Parameter("br2", 1 if self.in_operation else 0),
            ]

        elif self.branch_type == "Transformer":
            # Yes: from_bus and to_bus seem to be inverted
            return [
                Parameter("bus1", self.to_bus.name),
                Parameter("bus2", self.from_bus.name),
                Parameter(
                    "controlled_bus",
                    self.to_bus.name if self.has_OLTC else "' '",
                ),
                Parameter(
                    "R",
                    100
                    * change_base(self.R_pu, self.sys.base_MVA, self.Snom_MVA),
                ),
                Parameter(
                    "X",
                    100
                    * change_base(self.X_pu, self.sys.base_MVA, self.Snom_MVA),
                ),
                Parameter(
                    "B",
                    100
                    * change_base(
                        self.from_Y_pu.imag, self.sys.base_MVA, self.Snom_MVA
                    ),
                ),
                Parameter("N", self.n_pu * 100),
                Parameter("Snom_MVA", self.Snom_MVA),
                Parameter("n_first", 0),
                Parameter("n_last", 0),
                Parameter("n_pos", 0),
                Parameter("tol_v", 0),
                Parameter("v_des", 0),
                Parameter("breaker", 1),
            ]


class OLTC:
    """
    (Mainly) the mechanical part of on-load tap changers (OLTCs).

    This class abstracts the notion of increasing voltages, decreasing them, or
    letting the OLTC act in the direction it prefers.

    These objects are not meant to be printed to RAMSES and hence some
    parameters like the delays have been removed.
    """

    def __init__(
        self,
        trafo: Branch,
        positions_up: int,
        positions_down: int,
        step_pu: float,
        v_setpoint_pu: float,
        half_db_pu: float,
    ) -> None:
        """
        The controlled bus is the to_bus, since n is on the 'from' side.
        """

        attributes = vars()
        for key in attributes:
            setattr(self, key, attributes[key])

        # Infer parameters of the OLTC
        self.pos = 0  # position of the mechanism (probably wrong)
        self.nmin_pu = 1 - self.positions_down * self.step_pu
        self.nmax_pu = 1 + self.positions_up * self.step_pu
        self.controlled_bus = trafo.to_bus
        self.OLTC_controller = None

    def increase_voltages(self) -> bool:
        """
        Increase voltage of controlled bus and indicate whether it did.
        """

        if self.pos > -self.positions_down:
            self.trafo.n_pu -= self.step_pu
            self.pos -= 1
            return True
        else:
            return False

    def reduce_voltages(self) -> bool:
        """
        Reduce voltage of controlled bus and indicate whether it did.
        """

        if self.pos < self.positions_up:
            self.trafo.n_pu += self.step_pu
            self.pos += 1
            return True
        else:
            return False

    def act(self) -> bool:
        """
        Act in the direction prefered by the OLTC and indicate whether it did.

        The controlled bus is the to_bus, since n is on the 'from' side.

        This method's output is defined by increase_voltages() and
        reduce_voltages(). Returning this boolean is useful in the method
        match_power() of StaticSystem to ensure termination.
        """

        if self.controlled_bus.V_pu < self.v_setpoint_pu - self.half_db_pu:
            return self.increase_voltages()
        elif self.controlled_bus.V_pu > self.v_setpoint_pu + self.half_db_pu:
            return self.reduce_voltages()
        else:
            return False


class DERA(Injector):
    """
    DER_A model in RAMSES.
    """

    prefix: str = "INJEC DERA"

    def __init__(
        self,
        name: str,
        bus: Bus,
        P0_MW: float,
        Q0_Mvar: float,
        Snom_MVA: float,
        Tp: float = 0.02,
        fdbd1: float = -0.0006,  # Zero or negative number
        fdbd2: float = 0.0006,  # Zero or positive number
        Ddn: float = 20,
        Dup: float = 20,
        femin: float = -99,
        femax: float = 99,
        kig: float = 10,
        kpg: float = 0.1,
        Pmin: float = 0,
        Pmax: float = 1,
        dPmin: float = -99,
        dPmax: float = 99,
        Freq_flag: int = 0, # 1 for P-f control, 0 for P reference
        Pflag: int = 0,  # 1 for power factor reference, 0 to reactive power reference
        Pqflag: int = 1, # 1 for P priority, 0 for Q priority
        typeflag: int = 1, # 1 if generator, 0 if storage
        Tpord: float = 5,
        Trv: float = 0.02,
        vref0: float = 0, # the model sets its own reference voltage based on initial conditions
        dbd1: float = -99,  # Zero or negative number (IEEE 1547-2018 recommends -99, as in most applications DERs do not control voltage, e.g. -0.01)
        dbd2: float = 99,  # Zero or positive number (IEEE 1547-2018 recommends 99, as in most applications DERs do not control voltage, e.g. 0.01)
        kqv: float = 0, # Voltage control gain (e.g. 8)
        Iql1: float = -1,
        Iqh1: float = 1,
        Tiq: float = 0.02,
        Imax: float = 1.2,
        fl: float = 0.942,
        tfl: float = 0.16,
        fh: float = 1.03,
        tfh: float = 0.16,
        vl1: float = 0.6,
        tvl1: float = 0.16,
        vl0: float = 0.45,
        tvl0: float = 0.16,
        vh1: float = 1.15,
        tvh1: float = 0.16,
        vh0: float = 1.2,
        tvh0: float = 0.16,
        Trf: float = 0.1, # No idea where this value came from
        Vpr: float = 0.3,
        Tg: float = 0.02,
        rrpwr: float = 2,
        Vrfrac: float = 1.0, # No vintage DERs
        VtripFlag: float = 1.0, # 1 if tripping logic is enabled, 0 if not
        Tv: float = 0.02,
    ) -> None:
        attributes = vars()
        for key in attributes:
            setattr(self, key, attributes[key])

    def get_P(self) -> float:
        """
        Return constant P since a DER_A is considered a constant power source.
        """

        return self.P0_MW

    def get_Q(self) -> float:
        """
        Return constant Q since a DER_A is considered a constant power source.
        """

        return self.Q0_Mvar

    def get_pars(self) -> list[Parameter]:
        return [
            Parameter("bus", self.bus.name),
            Parameter("FP", 0),
            Parameter("FQ", 0),
            Parameter("P", self.get_P()),
            Parameter("Q", self.get_Q()),
            Parameter("Snom_MVA", self.Snom_MVA),
            Parameter("Tp", self.Tp),
            Parameter("fdbd1", self.fdbd1),
            Parameter("fdbd2", self.fdbd2),
            Parameter("Ddn", self.Ddn),
            Parameter("Dup", self.Dup),
            Parameter("femin", self.femin),
            Parameter("femax", self.femax),
            Parameter("kig", self.kig),
            Parameter("kpg", self.kpg),
            Parameter("Pmin", self.Pmin),
            Parameter("Pmax", self.Pmax),
            Parameter("dPmin", self.dPmin),
            Parameter("dPmax", self.dPmax),
            Parameter("Freq_flag", self.Freq_flag),
            Parameter("Pflag", self.Pflag),
            Parameter("Pqflag", self.Pqflag),
            Parameter("typeflag", self.typeflag),
            Parameter("Tpord", self.Tpord),
            Parameter("Trv", self.Trv),
            Parameter("vref0", self.vref0),
            Parameter("dbd1", self.dbd1),
            Parameter("dbd2", self.dbd2),
            Parameter("kqv", self.kqv),
            Parameter("Iql1", self.Iql1),
            Parameter("Iqh1", self.Iqh1),
            Parameter("Tiq", self.Tiq),
            Parameter("Imax", self.Imax),
            Parameter("fl", self.fl),
            Parameter("tfl", self.tfl),
            Parameter("fh", self.fh),
            Parameter("tfh", self.tfh),
            Parameter("vl1", self.vl1),
            Parameter("tvl1", self.tvl1),
            Parameter("vl0", self.vl0),
            Parameter("tvl0", self.tvl0),
            Parameter("vh1", self.vh1),
            Parameter("tvh1", self.tvh1),
            Parameter("vh0", self.vh0),
            Parameter("tvh0", self.tvh0),
            Parameter("Trf", self.Trf),
            Parameter("Vpr", self.Vpr),
            Parameter("Tg", self.Tg),
            Parameter("rrpwr", self.rrpwr),
            Parameter("Vrfrac", self.Vrfrac),
            Parameter("VtripFlag", self.VtripFlag),
            Parameter("Tv", self.Tv),
        ]


class INDMACH1(Injector):
    """
    Induction machine model INDMACH1 in RAMSES.
    """

    prefix: str = "INJEC INDMACH1"

    def __init__(
        self,
        name: str,
        bus: Bus,
        P0_MW: float,
        Q0_Mvar: float,
        Snom_MVA: float,
        RS: float,
        LLS: float,
        LSR: float,
        RR: float,
        LLR: float,
        H: float,
        A: float,
        B: float,
        LF: float,
    ) -> None:
        attributes = vars()
        for key in attributes:
            setattr(self, key, attributes[key])

    def get_P(self) -> float:
        """
        Return constant P since INDMACH1 is considered a constant-power load.

        The negative sign is because the load is negative injection.
        """

        return -self.P0_MW

    def get_Q(self) -> float:
        """
        Return constant Q since INDMACH1 is considered a constant-power load.

        The negative sign is because the load is negative injection.
        """

        return -self.Q0_Mvar

    def get_pars(self) -> list[Parameter]:
        return [
            Parameter("bus", self.bus.name),
            Parameter("FP", 0),
            Parameter("FQ", 0),
            Parameter("P", self.get_P()),
            Parameter("Q", self.get_Q()),
            Parameter("Snom_MVA", self.Snom_MVA),
            Parameter("RS", self.RS),
            Parameter("LLS", self.LLS),
            Parameter("LSR", self.LSR),
            Parameter("RR", self.RR),
            Parameter("LLR", self.LLR),
            Parameter("H", self.H),
            Parameter("A", self.A),
            Parameter("B", self.B),
            Parameter("LF", self.LF),
        ]
