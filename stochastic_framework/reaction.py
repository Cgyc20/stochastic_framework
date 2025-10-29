"""
Author: Charlie Cameron
Date: 29th October 2025
This code will basically add reactions to the stoichiometry.
"""
import numpy as np
import pandas as pd


class Reaction:


    def __init__(self):

        """
        This initialises the class object.
        """

        self.reaction_set = []



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

    def calculate_stoichiometry(self):
        """Calculate stoichiometry and return a labelled DataFrame."""

        self.number_of_reactions = len(self.reaction_set)
        if self.number_of_reactions == 0:
            raise ValueError("No reactions have been added to the system.")

        # Determine species present
        species_set = set()
        for reaction in self.reaction_set:
            species_set.update(reaction["reactants"].keys())
            species_set.update(reaction["products"].keys())

        self.species_list = sorted(species_set)
        self.number_of_species = len(self.species_list)

        # Initialise stoichiometric matrix
        self.stoichiometric_matrix = np.zeros(
            (self.number_of_species, self.number_of_reactions),
            dtype=int
        )

        # Fill matrix
        for j, reaction in enumerate(self.reaction_set):
            for species, coeff in reaction["reactants"].items():
                i = self.species_list.index(species)
                self.stoichiometric_matrix[i, j] -= coeff
            for species, coeff in reaction["products"].items():
                i = self.species_list.index(species)
                self.stoichiometric_matrix[i, j] += coeff

        # Create labels
        self.stoichiometric_species_labels = self.species_list
        self.stoichiometric_reaction_labels = [f"R{j+1}" for j in range(self.number_of_reactions)]

        # Convert to DataFrame ✅
        self.stoichiometric_df = pd.DataFrame(
            self.stoichiometric_matrix,
            index=self.stoichiometric_species_labels,
            columns=self.stoichiometric_reaction_labels
        )

        print(self.stoichiometric_df)
        print(self.stoichiometric_matrix)
    

    def print_reactions(self):
        """
        Print reactions in stochastic form.
        If reactants or products are empty, display 0.
        """

        self.number_of_reactions = len(self.reaction_set)
        if self.number_of_reactions == 0:
            raise ValueError("No reactions have been added to the system.")

        for idx, reaction in enumerate(self.reaction_set):
            # Reactants string
            if reaction["reactants"]:
                reactant_str = ' + '.join([f"{v}{k}" if v != 1 else f"{k}" for k, v in reaction["reactants"].items()])
            else:
                reactant_str = "0"

            # Products string
            if reaction["products"]:
                product_str = ' + '.join([f"{v}{k}" if v != 1 else f"{k}" for k, v in reaction["products"].items()])
            else:
                product_str = "0"

            print(f"Reaction {idx + 1}: {reactant_str} -> {product_str} (Rate: {reaction['reaction_rate']})")





           







Model = Reaction()

Model.add_reaction({"A": 1, "B": 1}, {"B": 2}, 0.1)
Model.add_reaction({},{"B":1},0.1)
Model.calculate_stoichiometry()
Model.print_reactions()