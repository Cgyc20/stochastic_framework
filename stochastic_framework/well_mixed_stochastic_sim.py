import numpy as np
from .reaction import Reaction
from tqdm import tqdm
import json


class WellMixedSSA:
    """
    Well-Mixed Stochastic Simulation Algorithm (Gillespie) for a Reaction system.
    No spatial structure - all reactions occur in a single, homogeneous volume.
    """

    def __init__(self, reaction_system: Reaction):
        """
        Parameters
        ----------
        reaction_system : Reaction
            An instance of the Reaction class containing reactions and stoichiometry.
        """
        self.reaction_system = reaction_system
        self.species_list = reaction_system.species_list
        self.n_species = len(self.species_list)
        self.species_index = reaction_system.species_index
        self.stoichiometric_matrix = reaction_system.stoichiometric_matrix
        self.number_of_reactions = reaction_system.number_of_reactions
        self.reaction_set = reaction_system.reaction_set

    def set_conditions(self,
                       volume: float,
                       total_time: float,
                       initial_conditions: np.ndarray,
                       timestep: float):
        """
        Set initial conditions and validate inputs.

        Parameters
        ----------
        volume : float
            Volume of the well-mixed compartment.
        total_time : float
            Total simulation time.
        initial_conditions : np.ndarray
            Array of initial molecule counts for each species (shape: (n_species,)).
        timestep : float
            Time step for recording snapshots.
        """
        if not isinstance(volume, float) or volume <= 0:
            raise ValueError("Volume must be a positive float")

        if not isinstance(total_time, float) or total_time <= 0:
            raise ValueError("Total time must be a positive float")

        if not isinstance(initial_conditions, np.ndarray):
            raise ValueError("Initial conditions must be a numpy array")

        if initial_conditions.shape != (self.n_species,):
            raise ValueError(f"Initial conditions must be of shape ({self.n_species},)")

        # Ensure initial_conditions are non-negative integers
        if not np.issubdtype(initial_conditions.dtype, np.integer):
            if np.all(initial_conditions >= 0):
                initial_conditions = initial_conditions.astype(int)
            else:
                raise ValueError("Initial conditions must be non-negative integers")

        if not isinstance(timestep, float) or timestep <= 0:
            raise ValueError("Timestep must be a positive float")

        # Store values
        self.volume = volume
        self.total_time = total_time
        self.initial_conditions = initial_conditions
        self.timestep = timestep
        self.timevector = np.arange(0, self.total_time, self.timestep)
        self.propensity_vector = np.zeros(self.number_of_reactions)

        print("All initial conditions are valid.")

        # Initialize results storage
        self._generate_results_array()

    def _generate_results_array(self):
        """
        Generates the results array for the system.
        Dimensions = (number_of_timepoints, n_species)
        """
        results = np.zeros((len(self.timevector), self.n_species), dtype=int)
        results[0, :] = self.initial_conditions
        self.results = results
        return results

    def _propensity_calculation(self, state: np.ndarray, propensity_vector: np.ndarray):
        """
        Calculates the propensity functions for each reaction in the well-mixed volume.

        Parameters
        ----------
        state : np.ndarray
            Current state vector of shape (n_species,) with molecule counts.
        propensity_vector : np.ndarray
            Array of shape (number_of_reactions,) to store propensities.

        Returns
        -------
        propensity_vector : np.ndarray
            Updated propensity vector.
        """
        assert state.shape == (self.n_species,), "State shape is incorrect"
        assert propensity_vector.shape == (self.number_of_reactions,), "Propensity vector shape is incorrect"

        for i, reaction in enumerate(self.reaction_set):
            reaction_type = reaction['reaction_type']
            reactant_indices = reaction['reactant_indices']
            rate = reaction['reaction_rate']

            if reaction_type == 'zero_order':
                # Constant production (rate scaled by volume)
                propensity_vector[i] = rate * self.volume

            elif reaction_type == 'first_order':
                idx = reactant_indices[0]
                propensity_vector[i] = rate * state[idx]

            elif reaction_type == 'second_order':
                if len(reactant_indices) == 1:
                    # A + A -> ... (same species)
                    idx = reactant_indices[0]
                    propensity_vector[i] = rate * state[idx] * (state[idx] - 1) / self.volume
                else:
                    # A + B -> ... (different species)
                    idx1, idx2 = reactant_indices
                    propensity_vector[i] = rate * state[idx1] * state[idx2] / self.volume
            else:
                raise ValueError(f"Unknown reaction type: {reaction_type}")

        return propensity_vector

    def _SSA_loop(self):
        """
        Run the well-mixed SSA simulation using the Gillespie algorithm.
        """
        t = 0.0
        old_time = t
        current_state = self.initial_conditions.copy()

        while t < self.total_time:
            # Calculate propensities
            propensity_vector = self._propensity_calculation(
                state=current_state,
                propensity_vector=self.propensity_vector
            )

            alpha0 = np.sum(propensity_vector)
            if alpha0 == 0:
                # No further reactions possible
                break

            # Draw random numbers
            r1, r2 = np.random.rand(2)

            # Time to next reaction
            tau = (1 / alpha0) * np.log(1 / r1)

            # Select which reaction occurs
            alpha_cum = np.cumsum(propensity_vector)
            reaction_index = np.searchsorted(alpha_cum, r2 * alpha0)

            # Update state according to stoichiometry
            stoichiometric_update = self.stoichiometric_matrix[:, reaction_index]
            current_state += stoichiometric_update

            # Update time
            old_time = t
            t += tau

            # Record state at appropriate time points
            ind_before = int(old_time / self.timestep)
            ind_after = int(t / self.timestep)

            if ind_after > ind_before:
                for time_index in range(ind_before + 1, min(ind_after + 1, len(self.timevector))):
                    self.results[time_index, :] = current_state

        return self.results

    def run_simulation(self, n_repeats: int):
        """
        Run the well-mixed SSA simulation multiple times and return the average.

        Parameters
        ----------
        n_repeats : int
            Number of simulation repeats.

        Returns
        -------
        results : np.ndarray
            Array of shape (number_of_timepoints, n_species) containing averaged simulation results.
        """
        summed_results = np.zeros((len(self.timevector), self.n_species), dtype=int)
        final_results = np.zeros((len(self.timevector), self.n_species), dtype=float)

        for _ in tqdm(range(n_repeats)):
            self._generate_results_array()  # Reset results for each repeat
            summed_results += self._SSA_loop()

        final_results = summed_results / n_repeats

        return final_results

    def save_simulation_data(self, filename: str, simulation_result: np.ndarray):
        """
        Save the well-mixed SSA simulation data, time vector, volume, and reaction info to a .npz file.

        Parameters
        ----------
        filename : str
            Full path where the file should be saved (including .npz extension).
        simulation_result : np.ndarray
            The simulation result to save (e.g., output from run_simulation).
        """
        # Convert reaction set into JSON-serializable format
        reaction_data = []
        for r in self.reaction_set:
            reaction_data.append({
                'reactants': r.get('reactants', {}),
                'products': r.get('products', {}),
                'reaction_type': r.get('reaction_type', ''),
                'reaction_rate': r.get('reaction_rate', 0.0)
            })

        # Save everything into a .npz file
        np.savez_compressed(
            filename,
            simulation_result=simulation_result,
            timevector=self.timevector,
            volume=self.volume,
            total_time=self.total_time,
            timestep=self.timestep,
            n_species=self.n_species,
            reaction_data=json.dumps(reaction_data)  # Save as JSON string
        )

        print(f"Simulation data successfully saved to {filename}")
