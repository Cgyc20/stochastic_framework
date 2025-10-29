from stochastic_framework import Reaction


Model = Reaction()

Model.add_reaction({"A": 1}, {"B": 1}, 0.5)
Model.add_reaction({"B": 1}, {"A": 1}, 0.6)
Model.add_reaction({},{"A":1},1.0)
Model.add_reaction({"B":1},{},0.1)

Model.print_reactions()
stoich_df = Model.show_stoichiometry()
print(stoich_df)