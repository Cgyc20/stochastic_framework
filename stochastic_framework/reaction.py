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

    def calculate_stoichiometry(self):
        """This will calculate the stoichiometry of the reaction system"""

        self.number_of_reactions = len(self.reaction_set)
        
        if self.number_of_reactions == 0:
            raise ValueError("No reactions have been added to the system.")

        else:
            print(f"Calculating stochiometry for a total of {self.number_of_reactions} reactions")

        #Check if we have one or two species present within the system
        species_set = set()
        for reaction in self.reaction_set:
            species_set.update(reaction["reactants"].keys())
            species_set.update(reaction["products"].keys())
        self.species_list = list(species_set)
        self.number_of_species = len(self.species_list)

        if self.number_of_species == 1:
            print("One species present in the system")
        elif self.number_of_species == 2:
            print("Two species present in the system")

        

           







Model = Reaction()

Model.add_reaction({"A": 1, "B": 1}, {"B": 1}, 0.1)
Model.add_reaction({},{"B":1},0.1)
Model.calculate_stoichiometry()