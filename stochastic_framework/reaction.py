"""
Author: Charlie Cameron
Date: 29th October 2025
This code will basically add reactions to the stoichiometry.
"""




class Reaction:


    def __init__(self):

        """
        This initialises the class object.
        """

        self.reaction_set = []
        self.stoichiometry = []


    def add_reaction(self, reactants: dict, products: dict, reaction_rate: float):
        """Adds a reactions to the system"""

        assert isinstance(reactants, dict), "Reactants must be a dictionary"
        assert isinstance(products, dict), "Products must be a dictionary"
        assert isinstance(reaction_rate, float), "Reaction rate must be a float"


        unique_species = set(reactants.keys()).union(set(products.keys()))
        
        if len(unique_species) == 0:
            raise ValueError("At least one reactant or product must be specified.")
        elif len(unique_species) == 1:
            print(f"This reaction involves only one species")
        elif len(unique_species) == 2:
            print(f"This reaction involves two species")
        else:
            raise ValueError("This can only handle two species currently.")

        number_of_reacting_molecules = sum(list(reactants.values()))
        
        if number_of_reacting_molecules < 0:
            raise ValueError("At least one reactant molecule must be specified.")
        elif number_of_reacting_molecules ==0 :
            reaction_type = 'zero_order'
            print("A zeroth order reaction")
        elif number_of_reacting_molecules == 1:
            reaction_type = 'first_order'
            print("A first order reaction")
        elif number_of_reacting_molecules == 2:
            reaction_type = 'second_order'
            print("A second order reaction")
        else:
            raise ValueError("This can only handle reactions with up to two reacting molecules currently.")

        self.reaction_set.append({
            "reactants": reactants,
            "products": products,
            "reaction_rate": reaction_rate,
            "reaction_type": reaction_type
        })

    # def calculate_stoichiometry():
    #     """This will calculate the stoichiometry of the reaction system"""








Model = Reaction()

Model.add_reaction({"A": 1, "B": 1}, {"B": 1}, 0.1)
Model.add_reaction({},{"B":1},0.1)
