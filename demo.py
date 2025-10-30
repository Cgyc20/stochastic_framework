from stochastic_framework import Reaction, SSA
import numpy as np

ReactionData = Reaction()

ReactionData.add_reaction({"A": 2}, {"B": 1}, 0.5)
ReactionData.add_reaction({"B": 1}, {"A": 1}, 0.6)
ReactionData.add_reaction({},{"A":1},1.0)
ReactionData.add_reaction({"B":1},{},0.1)

ReactionData.print_reactions()
stoich_df = ReactionData.show_stoichiometry()
print(stoich_df)

SSA_class = SSA(ReactionData)

print(SSA_class.stoichiometric_matrix)

SSA_class.set_conditions(
    n_compartments=5,
    domain_length=10.0,
    total_time=100.0,
    initial_conditions=np.array([[5, 0, 1,2,5 ],
                                [30, 10, 5, 1,2],
                                    ]),

    timestep=0.1,
    Macroscopic_diffusion_rates=[0.01, 0.02]
)

SSA_class._propensity_calculation(
    dataframe=SSA_class.initial_conditions,
    propensity_vector=np.zeros(SSA_class.n_compartments*len(SSA_class.species_list) + SSA_class.n_compartments*SSA_class.reaction_system.number_of_reactions)
)

average_output = SSA_class.run_simulation(n_repeats=5)

SSA_class.save_simulation_data(
    filename="/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/SSA_data.npz",
    simulation_result=average_output
) 

print(SSA_class.space)