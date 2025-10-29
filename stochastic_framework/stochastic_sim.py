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
        # Validate initial_conditions type
        if not isinstance(initial_conditions, np.ndarray):
            raise ValueError("Initial conditions must be a numpy array")

        if initial_conditions.shape != (number_of_compartments, len(self.species_list)):
            raise ValueError(f"Initial conditions must be of shape ({number_of_compartments}, {len(self.species_list)})")

        # Ensure initial_conditions are non-negative integers
        if not np.issubdtype(initial_conditions.dtype, np.integer):
            if np.all(initial_conditions >= 0):
                initial_conditions = initial_conditions.astype(int)
            else:
                raise ValueError("Initial conditions must be non-negative integers")

        self.initial_conditions = initial_conditions
        
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
        self.timevector = np.arange(0,self.total_time, self.timestep)
        self.h = self.domain_length / self.number_of_compartments
        self.space = np.linspace(0,self.domain_length-self.h,self.number_of_compartments)

        print("All initial conditions are valid.")


        initial_tensor = self._generate_dataframes()


        self.jump_rate_list = [macroscopic_rate/(self.h**2) for macroscopic_rate in self.Macroscopic_diffusion_rates]



    def _generate_dataframes(self):
        """
        Generates the dataframes for the system. It is going to be a three dimensional tensor.
        Dimensions = (number_of_timepoints, number_of_compartments, number_of_species)
        """

        tensor = np.zeros((len(self.timevector), self.number_of_compartments, len(self.species_list)), dtype = int)

        tensor[0,:,:] = self.initial_conditions
        self.tensor = tensor
        return tensor
    
    def _propensity_calculation(self,
                                dataframe: np.ndarray,
                                propensity_vector: np.ndarray):
        """
        Calculates the propensity_vector functions for each reaction in each compartment.
        This will also take into account the diffusion propensity_vector. 
        
        Parameters:
         Dataframe: A np.ndarray dataframe of shape (self.number_of_compartments, number_of_species)
        
        propensity_vector: A np.ndarray array of shape (self.number_of_compartments*number_of_species + self.number_of_compartments*number_of_reactions,)

        Returns: The updated Propensity function:
        
        """

        assert dataframe.shape == (self.number_of_compartments, len(self.species_list)), "Dataframe shape is incorrect"
        assert propensity_vector.shape == (self.number_of_compartments*len(self.species_list) + self.number_of_compartments*self.reaction_system.number_of_reactions,), "Propensity vector shape is incorrect"
        #assert that the propensity elements are floats

        
        #First we will do the Movement (diffusion propensities)

        for species_index in range(len(self.species_list)):
            corresponding_jump_rate = self.jump_rate_list[species_index]

            propensity_vector[species_index*self.number_of_compartments:(species_index+1)*self.number_of_compartments] = corresponding_jump_rate * dataframe[:,species_index]*2.0

        # Now we are going to 

        
        
        

