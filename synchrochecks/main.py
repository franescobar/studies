import pf_static
import records
from collections.abc import Sequence
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import numpy as np


class OptimizableSystem(pf_static.StaticSystem):
    pass

    def define_redispatchable_generators(
        self, generator_names: Sequence[str]
    ) -> None:
        """
        Define the genenerators whose active power can be redispatched.
        """

        self.redispatchable_generators: list[records.Generator] = []

        for generator_name in generator_names:
            # We first fetch the associated object:
            generator = self.gen_dict[generator_name]

            # A redispatchable generator cannot be connected to the slack
            # bus, as otherwise we would loose a degree of freedom.
            if generator.bus is self.slack:
                raise ValueError("Cannot redispatch slack generator.")

            # We then store the generator and store its initial power output
            # to use it when evaluating the cost function.
            self.redispatchable_generators.append(generator)
            generator.PG_MW_0 = generator.PG_MW

    def specify_synchrocheck(
        self,
        line_name: str,
        bus_name: str,
        V_pu_tol: float,
        theta_degrees_tol: float,
    ) -> None:
        """
        Specify the synchrocheck under study, as well as its settings.
        """

        # To simulate an open line, we will add a fictitious bus that represents
        # the breaker pole that is opposite to the specified bus. The
        # optimization will then try to bring the voltages of those two buses
        # closer together.

        line = self.line_dict[line_name]
        bus = self.bus_dict[bus_name]

        self.SC_bus = bus
        self.fictitious_bus = self.add_PQ(
            PL=0, QL=0, name=f"open end of {line_name}", base_kV=bus.base_kV
        )

        if bus is line.from_bus:
            line.from_bus = self.fictitious_bus
        elif bus is line.to_bus:
            line.to_bus = self.fictitious_bus
        else:
            raise ValueError("The line does not touch the specified bus.")

        # We then save the settings of the synchrocheck.
        self.V_pu_tol = V_pu_tol
        self.theta_degrees_tol = theta_degrees_tol

    def set_generator_limits(
        self, generator_name: str, PG_MW_min: float, PG_MW_max: float
    ) -> None:
        """
        Set the active power limits of a generator.
        """

        self.gen_dict[generator_name].PG_MW_min = PG_MW_min
        self.gen_dict[generator_name].PG_MW_max = PG_MW_max

    def set_bus_limits(
        self, bus_name: str, V_pu_min: float, V_pu_max: float
    ) -> None:
        """
        Set the voltage limits of a bus.
        """

        self.bus_dict[bus_name].V_pu_min = V_pu_min
        self.bus_dict[bus_name].V_pu_max = V_pu_max

    def print_problem_formulation(self) -> None:
        """
        Display the OPF problem formulation.

        The message has the form:

            You want to solve the following OPF problem:

                minimize the sum of: (PG of j - initial PG of j)^2
                over all generators j

                subject to:

                    * PG_min of j <= PG of j <= PG_max of j  for every gen. j,
                    * V_min of i <= V of i <= V_max of i     for every bus  i,
                    * abs(V_pu difference across breaker) <= V_pu_tol,
                    * abs(theta difference across breaker) <= theta_degrees_tol,
                    * and, implicitly, the power flow equations.
        """

        message = "\nYou want to solve the following OPF problem:\n\n"

        message += "    minimize the sum of: (PG of j - initial PG of j)^2\n"
        message += "    over all generators j\n\n"

        message += "    subject to:\n\n"

        for generator in self.redispatchable_generators:
            message += (
                f"        {generator.PG_MW_min:.1f} MW "
                f"<= PG of {generator.name} "
                f"<= {generator.PG_MW_max:.1f} MW\n"
            )

        message += "\n"
        for bus in self.buses:
            message += (
                f"        {bus.V_pu_min:.2f} pu "
                f"<= V of {bus.name} "
                f"<= {bus.V_pu_max:.2f}\n"
            )

        message += "\n"
        message += (
            f"        "
            f"abs(V of {self.fictitious_bus.name} "
            f"- "
            f"V of {self.SC_bus.name}) <= {self.V_pu_tol:.2f} pu\n"
        )
        message += (
            f"        "
            f"abs(theta of {self.fictitious_bus.name} "
            f"- "
            f"theta of {self.SC_bus.name}) "
            f"<= {self.theta_degrees_tol:.2f} degrees\n"
        )
        message += "\n"
        message += "        and, implicitly, the power flow equations. ;)\n\n"

        print(message)

        input("Press Enter to continue...")

    def find_optimal_redispatch(self, power_flow_tol: float = 1e-9) -> None:
        """
        Solve the OPF problem and display optimization results.
        """

        # Print the problem formulation:
        self.print_problem_formulation()

        # Define the cost function:
        def fun(x: np.ndarray) -> float:
            cost = 0
            for new_PG_MW, generator in zip(x, self.redispatchable_generators):
                cost += (new_PG_MW - generator.PG_MW_0) ** 2

            return cost

        # Define the linear constraints on generation output, i.e. lb <= Ax <= ub:
        lb = np.array(
            [
                generator.PG_MW_min
                for generator in self.redispatchable_generators
            ]
        )
        ub = np.array(
            [
                generator.PG_MW_max
                for generator in self.redispatchable_generators
            ]
        )
        A = np.eye(len(self.redispatchable_generators))
        LC = LinearConstraint(A=A, lb=lb, ub=ub)

        # Define the nonlinear constraints on voltage magnitude and angle:
        def voltage_magnitudes(x: np.ndarray) -> np.ndarray:
            """
            Return the voltage magnitudes of all buses.
            """

            # We first update the power output of the redispatchable generators:
            for new_PG_MW, generator in zip(x, self.redispatchable_generators):
                generator.PG_MW = new_PG_MW

            # We then run the power flow:
            self.run_pf(tol=power_flow_tol)

            # We finally return the voltage magnitudes:
            return np.array([bus.V_pu for bus in self.buses])

        def differences_across_breaker() -> np.ndarray:
            """
            Return the voltage magnitude and angle differences across the breaker.

            We take advantage of the fact that the power flow has already been
            run when this function is called.
            """

            delta_V = np.abs(self.fictitious_bus.V_pu - self.SC_bus.V_pu)
            delta_theta_degrees = np.abs(
                np.rad2deg(self.fictitious_bus.theta_radians)
                - np.rad2deg(self.SC_bus.theta_radians)
            )

            # We then return the differences:
            return np.array([delta_V, delta_theta_degrees])

        def NLC_fun(x: np.ndarray) -> np.ndarray:
            """
            This simply stacks the outputs of the two functions above.
            """

            mag = voltage_magnitudes(x)
            diff = differences_across_breaker()

            return np.concatenate([mag, diff])

        # The last two numbers are the bounds for the differences across the
        # breaker:
        lb = np.array([bus.V_pu_min for bus in self.buses] + [0, 0])
        ub = np.array(
            [bus.V_pu_max for bus in self.buses]
            + [self.V_pu_tol, self.theta_degrees_tol]
        )

        NLC = NonlinearConstraint(fun=NLC_fun, lb=lb, ub=ub)

        # Define the initial guess (initial dispatch):
        x0 = np.array(
            [generator.PG_MW for generator in self.redispatchable_generators]
        )

        # Solve the problem (I just played around with the solver options until
        # it worked)
        res = minimize(
            x0=x0,
            fun=fun,
            constraints=[LC, NLC],
            tol=1e-4,
            method="SLSQP",
            options={
                "ftol": 1e-4,
                "eps": 1e-3,
            },
        )

        # Print the results:
        self.print_redispatch()

        # Display the new operating point:
        print("\nThe new operating point is:")
        print(self)

    def print_redispatch(self) -> None:
        """
        Display the redispatch results.
        """

        # Print the changes: PG_MW_0 -> PG_MW (delta = PG_MW - PG_MW_0)
        print("\nThe redispatch is as follows:\n\n")
        for generator in self.redispatchable_generators:
            print(
                f"{generator.name}: "
                f"{generator.PG_MW_0:.1f} MW -> {generator.PG_MW:.1f} MW "
                f"(delta = {generator.PG_MW - generator.PG_MW_0:.1f} MW)"
            )

        # Print the magnitude and angle differences across the breaker:
        delta_V = np.abs(self.fictitious_bus.V_pu - self.SC_bus.V_pu)
        delta_theta_degrees = np.abs(
            np.rad2deg(self.fictitious_bus.theta_radians)
            - np.rad2deg(self.SC_bus.theta_radians)
        )
        print(
            f"\nVoltage magnitude difference across breaker: {delta_V:.2f} pu"
        )

        print(
            f"Voltage angle difference across breaker: "
            f"{delta_theta_degrees:.2f} degrees"
        )


if __name__ == "__main__":
    # We test the code with the Nordic test system, which gives us enough
    # degrees of freedom to play with.

    nordic = OptimizableSystem.import_ARTERE(
        filename="lf_A.dat",
        system_name="Nordic Test System",
        base_MVA=100,
        use_injectors=True,
    )

    # We then assume that the line 1011-1013 is open and that we want to close
    # the circuit breaker at bus 1013. We choose this line because it is not
    # critical and will not give convergence issues. Regarding the synchrocheck,
    # we assume that it tolerates a voltage magnitude difference of 5% and a
    # voltage angle difference of 10 degrees.  These are the values currently
    # used by the Costa Rican transmission system operator (ICE).

    nordic.specify_synchrocheck(
        line_name="1011-1013",
        bus_name="1013",
        V_pu_tol=0.1,
        theta_degrees_tol=10,
    )

    nordic.run_pf()
    print(nordic)
    input(f"Showing initial operating point. Press Enter to continue...")

    # We then choose the redispatchable generators. For simplicity, we choose
    # all of them except for the slack generator (g20). This results in
    # 19 redispatchable generators: g1, g2, ..., g19. Furthermore, we assume
    # that they are operating at 80 % of their maximum power output, and that
    # they can be redispatched between 60 % and 100 % of their maximum power.

    generator_names = [f"g{i}" for i in range(1, 20)]
    nordic.define_redispatchable_generators(generator_names=generator_names)

    for generator_name in generator_names:
        # Fetch the generator:
        generator = nordic.gen_dict[generator_name]
        # Infer power limits according to the assumptions above:
        current_output = generator.PG_MW
        PG_MW_max = current_output / 0.8
        PG_MW_min = 0.6 * PG_MW_max
        # Set the limits:
        nordic.set_generator_limits(
            generator_name=generator_name,
            PG_MW_min=PG_MW_min,
            PG_MW_max=PG_MW_max,
        )

    # Finally, we set the voltage limits of the buses. As per Costa Rican
    # regulations, we assume 0.9 pu - 1.1 pu for transmission buses and 0.95 pu
    # - 1.05 pu for distribution buses. A bus belongs to the transmission
    # network (for our purposes) if it does not feed any load. (The
    # expression below is ugly, sorry.)
    distribution_buses = {
        bus
        for bus in nordic.buses
        if any(
            inj.bus is bus
            for inj in nordic.injectors
            if isinstance(inj, records.Load)
        )
    }
    transmission_buses = set(nordic.buses) - distribution_buses

    for bus in distribution_buses:
        nordic.set_bus_limits(bus_name=bus.name, V_pu_min=0.95, V_pu_max=1.05)

    for bus in transmission_buses:
        nordic.set_bus_limits(bus_name=bus.name, V_pu_min=0.9, V_pu_max=1.1)

    # We can now run the redispatch:
    nordic.find_optimal_redispatch(power_flow_tol=1e-9)
