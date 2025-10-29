import numpy as np
from .reaction import Reaction



class SSA:
    """
    Stochastic Simulation Algorithm (Gillespie) for a Reaction system.
    """

    def __init__(self, reaction_system: Reaction):
        """
        Parameters
        ----------
        reaction_system : Reaction
            An instance of the Reaction class containing reactions and stoichiometry.
        initial_counts : dict
            Dictionary of species initial counts (integers).
        """
        self.reaction_system = reaction_system
        self.species_list = reaction_system.species_list
        self.stoichiometric_matrix = reaction_system.stoichiometric_matrix


    def set_conditions(self, 
                   number_of_compartments: int,
                   domain_length: float,
                   total_time: float,
                   initial_conditions: np.ndarray,
                   timestep: float,
                   Macroscopic_diffusion_rates: list):
        
        """Set initial conditions and validate inputs."""
        
        if not isinstance(number_of_compartments, int) or number_of_compartments <= 0:
            raise ValueError("Number of compartments must be positive")
        
        if not isinstance(domain_length, float):
            raise ValueError("Domain length must be a float")
        
        if not isinstance(total_time, float):
            raise ValueError("Total time must be a float")
        
        if not isinstance(initial_conditions, np.ndarray):
            raise ValueError("Initial conditions must be a numpy array")
        
        if initial_conditions.shape != (number_of_compartments, len(self.species_list)):
            raise ValueError(f"Initial conditions must be of shape ({number_of_compartments}, {len(self.species_list)})")
        
        if not isinstance(timestep, float):
            raise ValueError("Timestep must be a float")
        
        if not isinstance(Macroscopic_diffusion_rates, list):
            raise ValueError("Diffusion rates must be a list")
        
        if len(Macroscopic_diffusion_rates) != len(self.species_list):
            raise ValueError("Diffusion rates list length must match number of species")
        
        for rate in Macroscopic_diffusion_rates:
            if not isinstance(rate, float):
                raise ValueError("Each diffusion rate must be a float")

        # If all checks pass, store values
        self.number_of_compartments = number_of_compartments
        self.domain_length = domain_length
        self.total_time = total_time
        self.initial_conditions = initial_conditions
        self.timestep = timestep
        self.Macroscopic_diffusion_rates = Macroscopic_diffusion_rates

        print("All initial conditions are valid.")

    

        

