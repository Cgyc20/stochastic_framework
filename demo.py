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
    number_of_compartments=3,
    total_time=100.0,
    initial_conditions=np.array([[50, 0],
                                    [30, 10],
                                    [20, 5]]),

    timestep=0.1,
    Macroscopic_diffusion_rates=[0.01, 0.02]
)