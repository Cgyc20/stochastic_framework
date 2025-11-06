import numpy as np
from .reaction import Reaction
from tqdm import tqdm
import json

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
        self.n_species = len(self.species_list)
    
        self.species_index = reaction_system.species_index
        self.stoichiometric_matrix = reaction_system.stoichiometric_matrix
        self.number_of_reactions = reaction_system.number_of_reactions
        self.reaction_set = reaction_system.reaction_set
      
    

    def set_conditions(self, 
                   n_compartments: int,
                   domain_length: float,
                   total_time: float,
                   initial_conditions: np.ndarray,
                   timestep: float,
                   Macroscopic_diffusion_rates: list,
                   boundary_conditions:str):
        
        """Set initial conditions and validate inputs."""
        
        if not isinstance(n_compartments, int) or n_compartments <= 0:
            raise ValueError("Number of compartments must be positive")
        
        if not isinstance(domain_length, float):
            raise ValueError("Domain length must be a float")
        
        if not isinstance(total_time, float):
            raise ValueError("Total time must be a float")
        # Validate initial_conditions type
        if not isinstance(initial_conditions, np.ndarray):
            raise ValueError("Initial conditions must be a numpy array")

        if initial_conditions.shape != (self.n_species, n_compartments):
            raise ValueError(f"Initial conditions must be of shape ({self.n_species}, {n_compartments})")

        # Ensure initial_conditions are non-negative integers
        if not np.issubdtype(initial_conditions.dtype, np.integer):
            if np.all(initial_conditions >= 0):
                initial_conditions = initial_conditions.astype(int)
            else:
                raise ValueError("Initial conditions must be non-negative integers")

        self.initial_conditions = initial_conditions
        
      
        if not isinstance(timestep, float):
            raise ValueError("Timestep must be a float")
        
        if not isinstance(Macroscopic_diffusion_rates, list):
            raise ValueError("Diffusion rates must be a list")
        
        if len(Macroscopic_diffusion_rates) != self.n_species:
            raise ValueError("Diffusion rates list length must match number of species")
        
        for rate in Macroscopic_diffusion_rates:
            if not isinstance(rate, float):
                raise ValueError("Each diffusion rate must be a float")
        

        possible_boundary_strings = ['periodic', 'zero-flux']

        if not isinstance(boundary_conditions, str):
            raise ValueError("Boundary conditions must be a string")
        if boundary_conditions not in possible_boundary_strings:
            raise ValueError(f"Boundary conditions must be one of {possible_boundary_strings}")
        

        # If all checks pass, store values
        self.n_compartments = n_compartments
        self.domain_length = domain_length
        self.total_time = total_time
        self.initial_conditions = initial_conditions
        self.timestep = timestep
        self.Macroscopic_diffusion_rates = Macroscopic_diffusion_rates
        self.timevector = np.arange(0,self.total_time, self.timestep)
        self.h = self.domain_length / self.n_compartments
        self.space = np.linspace(0,self.domain_length-self.h,self.n_compartments)
        self.boundary_conditions = boundary_conditions
        self.propensity_vector = np.zeros(self.n_compartments*self.n_species + self.n_compartments*self.reaction_system.number_of_reactions)
        print("All initial conditions are valid.")


        initial_tensor = self._generate_dataframes()


        self.jump_rate_list = [macroscopic_rate/(self.h**2) for macroscopic_rate in self.Macroscopic_diffusion_rates]



    def _generate_dataframes(self):
        """
        Generates the dataframes for the system. It is going to be a three dimensional tensor.
        Dimensions = (number_of_timepoints, n_compartments, number_of_species)
        """

        tensor = np.zeros((len(self.timevector),  self.n_species, self.n_compartments), dtype = int)

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
         Dataframe: A np.ndarray dataframe of shape (self.n_compartments, number_of_species)
        
        propensity_vector: A np.ndarray array of shape (self.n_compartments*number_of_species + self.n_compartments*number_of_reactions,)

        Returns: The updated Propensity function:
        
        """

        assert dataframe.shape == (self.n_species, self.n_compartments), "Dataframe shape is incorrect"
        assert propensity_vector.shape == (self.n_compartments*self.n_species + self.n_compartments*self.reaction_system.number_of_reactions,), "Propensity vector shape is incorrect"
        #assert that the propensity elements are floats

        
        #First we will do the Movement (diffusion propensities)

        for species_index in range(self.n_species):
            corresponding_jump_rate = self.jump_rate_list[species_index]

            propensity_vector[species_index*self.n_compartments:(species_index+1)*self.n_compartments] = corresponding_jump_rate * dataframe[species_index, :]*2.0

        # Now we are going to run the reactions, from the reaction list. Note that we have self.number_of_reactions total reactions so we need to iterate through this list.

        #Note that for this we are going to fill up the propensities vectors from
        #Propensity[self.n_compartments*number_of_species: self.n_compartments*number_of_species+ self.n_compartments*number_of_reactions, ]
    
        # print("Species_index list", self.species_index)

        # print("Species list:", self.species_list)
        
        for i, reaction in enumerate(self.reaction_set):
            start = self.n_compartments * self.n_species + i * self.n_compartments
            end = start + self.n_compartments

            reaction_type = reaction['reaction_type']
            reactant_indices = reaction['reactant_indices']
            rate = reaction['reaction_rate']

            if reaction_type == 'zero_order':
                # Constant production
                propensity_vector[start:end] = rate * self.h

            
            elif reaction_type == 'first_order':
                idx = reactant_indices[0]
                propensity_vector[start:end] = rate*dataframe[idx,:]

            elif reaction_type == 'second_order':
                if len(reactant_indices) == 1: #Then its a single species reaction.
                    
                    idx = reactant_indices[0]
                    propensity_vector[start:end] = rate*dataframe[idx,:]*(dataframe[idx,:]-1)/self.h

                else:
                    #A + B
                    idx1, idx2 = reactant_indices
                    propensity_vector[start:end] = rate*dataframe[idx1, :]*dataframe[idx2,:]/self.h
            else:
                raise ValueError(f"Unknown reaction type {reaction_type}")
            
        return propensity_vector
    


    def _SSA_loop(self):
        """
        Run the SSA simulation.
        """
        
        t = 0.0
        old_time = t

        current_frame = self.initial_conditions.copy()

        while t<self.total_time:

            propensity_vector = self._propensity_calculation(
                dataframe=current_frame,
                propensity_vector=self.propensity_vector
            )

            alpha0 = np.sum(propensity_vector)
            if alpha0 == 0:
                #No further action
                break

            r1, r2, r3 = np.random.rand(3)
            tau = (1/alpha0)*np.log(1/r1)
            alpha_cum = np.cumsum(propensity_vector)
            index = np.searchsorted(alpha_cum, r2*alpha0)
            compartment_index = index % self.n_compartments

            # We first execute the diffusion reactions for each species of the model.
            
            if index < self.n_species*self.n_compartments:
                #Then we get diffusion.
                species_index = index // self.n_compartments
                
                if self.boundary_conditions == 'periodic':
                    if r3 < 0.5:
                        #Move left
                        current_frame[species_index, compartment_index] -= 1
                        current_frame[species_index, (compartment_index - 1)%self.n_compartments] += 1
                    else:
                        #Move right
                        current_frame[species_index, compartment_index] -= 1
                        current_frame[species_index, (compartment_index + 1)%self.n_compartments] += 1
                
                elif self.boundary_conditions == 'zero-flux':


                    if r3 < 0.5:
                        #Move left
                        if compartment_index == 0:
                            #Reflective boundary
                            current_frame[species_index, compartment_index] -=1 
                            current_frame[species_index, compartment_index+1] += 1

                        elif compartment_index == self.n_compartments-1:
                            #Reflective boundary
                            current_frame[species_index, compartment_index] -=1 
                            current_frame[species_index, compartment_index-1] += 1
                    
                        else:
                            current_frame[species_index, compartment_index] -= 1
                            current_frame[species_index, compartment_index - 1] += 1
                    else:
                        #Move right
                        if compartment_index == self.n_compartments - 1 or compartment_index == 0:
                            #Reflective boundary
                            pass
                        else:
                            current_frame[species_index, compartment_index] -= 1
                            current_frame[species_index, compartment_index + 1] += 1


            else:
                # we will have a reaction occuring (not diffusion)

                reaction_index = (index - self.n_species*self.n_compartments)//self.n_compartments

                stoichiometric_update = self.stoichiometric_matrix[:, reaction_index]
                current_frame[:, compartment_index] += stoichiometric_update
            old_time = t
            t += tau

            ind_before = int(old_time / self.timestep)
            ind_after = int(t / self.timestep)

            if ind_after > ind_before:
                for time_index in range(ind_before + 1, min(ind_after + 1, len(self.timevector))):
                    self.tensor[time_index, :, :] = current_frame

        return self.tensor
    

    def run_simulation(self, 
                       n_repeats: int):
        
        """
        Run the SSA simulation multiple times.
        Parameters:
        n_repeats : int
            Number of simulation repeats.
        Returns:
        -------
        results : np.ndarray
            Array of shape (n_repeats, number_of_timepoints, n_species, n_compartments)
            containing the simulation results.
        """

        summed_dataframe = np.zeros((len(self.timevector), self.n_species, self.n_compartments), dtype=int)
        final_dataframe = np.zeros((len(self.timevector), self.n_species, self.n_compartments), dtype=float)

        for _ in tqdm(range(n_repeats)):
            self._generate_dataframes()  # Reset tensor for each repeat
            summed_dataframe += self._SSA_loop()

        
        final_dataframe += summed_dataframe / n_repeats

        return final_dataframe
    

    def save_simulation_data(self, filename: str, simulation_result: np.ndarray):
        """
        Save the SSA simulation data, time vector, space, and reaction info to a single .npz file.

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
            space=self.space,
            domain_length = self.domain_length,
            total_time = self.total_time,
            timestep = self.timestep,
            h = self.h,
            n_species=self.n_species,
            n_compartments=self.n_compartments,
            jump_rates=np.array(self.jump_rate_list),
            reaction_data=json.dumps(reaction_data)  # Save as JSON string
        )

        print(f"Simulation data successfully saved to {filename}")

        

    





         
    

        
        
        

