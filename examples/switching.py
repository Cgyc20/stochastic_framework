from stochastic_framework import Reaction, SSA
import numpy as np



def main():


    ReactionData = Reaction()

    ReactionData.add_reaction({"A": 1}, {"B": 1}, 0.1)
    ReactionData.add_reaction({"B": 1}, {"A": 1}, 1.0)
    ReactionData.add_reaction({"B": 1}, {}, 0.2)


    ReactionData.print_reactions()
    stoich_df = ReactionData.show_stoichiometry()
    print(stoich_df)

    SSA_class = SSA(ReactionData)

    print(SSA_class.stoichiometric_matrix)

    n_compartments = 40
    A_initial = np.zeros(n_compartments, dtype=int) 
    B_initial = np.zeros(n_compartments, dtype=int)

    A_initial[0:10] = 100
    B_initial[-10:-1] = 100
    initial_conditions = np.zeros((2, n_compartments), dtype=int)
    initial_conditions[0, :] = A_initial
    initial_conditions[1, :] = B_initial

    SSA_class.set_conditions(
        n_compartments=n_compartments,
        domain_length=10.0,
        total_time=100.0,
        initial_conditions=initial_conditions,
        timestep=0.1,
        Macroscopic_diffusion_rates=[0.05, 0.05]
    )

    SSA_class._propensity_calculation(
        dataframe=SSA_class.initial_conditions,
        propensity_vector=np.zeros(SSA_class.n_compartments*len(SSA_class.species_list) + SSA_class.n_compartments*SSA_class.reaction_system.number_of_reactions)
    )

    average_output = SSA_class.run_simulation(n_repeats=5)

    SSA_class.save_simulation_data(
        filename="/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/SSA_data.npz",
        simulation_result=average_output
    ) 

    print(SSA_class.space)

if __name__ == "__main__":
    main()