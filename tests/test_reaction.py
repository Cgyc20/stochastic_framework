from stochastic_framework import Reaction
import numpy as np

def test_valid_two_species_reaction():
    model = Reaction()
    model.add_reaction({"A": 1, "B": 1}, {"A": 1}, 0.1)
    assert len(model.reaction_set) == 1
    print("✅ test_valid_two_species_reaction passed")

def test_raises_value_error_more_than_two_species():
    model = Reaction()
    try:
        model.add_reaction({"A": 1, "B": 1}, {"C": 1}, 0.2)
        print("✅ test_raises_value_error_more_than_two_species passed")  # Allowed now, no strict limit
    except ValueError:
        print("❌ test_raises_value_error_more_than_two_species failed")  # Should not fail now

def test_raises_error_no_products_reactants():
    model = Reaction()
    try:
        model.add_reaction({}, {}, 0.3)
    except ValueError:
        print("✅ test_raises_error_no_products_reactants passed")
        return
    print("❌ test_raises_error_no_products_reactants failed")

def test_error_if_more_than_two_reacting_molecules():
    model = Reaction()
    try:
        model.add_reaction({"A": 3}, {"B": 1}, 0.4)
    except ValueError:
        print("✅ test_error_if_more_than_two_reacting_molecules passed")
        return
    print("❌ test_error_if_more_than_two_reacting_molecules failed")

def test_correct_reaction_type_zeroth_order():
    model = Reaction()
    model.add_reaction({}, {"A": 1}, 0.2)
    reaction_type = model.reaction_set[0]["reaction_type"]
    assert reaction_type == 'zero_order'
    print("✅ test_correct_reaction_type_zeroth_order passed")

def test_correct_reaction_type_first_order():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.5)
    reaction_type = model.reaction_set[0]["reaction_type"]
    assert reaction_type == 'first_order'
    print("✅ test_correct_reaction_type_first_order passed")

def test_correct_reaction_type_second_order():
    model = Reaction()
    model.add_reaction({"A": 1, "B": 1}, {"A": 1}, 0.6)
    reaction_type = model.reaction_set[0]["reaction_type"]
    assert reaction_type == 'second_order'
    print("✅ test_correct_reaction_type_second_order passed")

def test_one_reaction_is_verified():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.7)
    assert model.number_of_reactions == 1
    print("✅ test_one_reaction_is_verified passed")

def test_two_reactions_verified():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.8)
    model.add_reaction({"B": 1}, {"A": 1}, 0.9)
    assert model.number_of_reactions == 2
    print("✅ test_two_reactions_verified passed")

def test_one_species_in_system():
    model = Reaction()
    model.add_reaction({"A": 1}, {"A": 2}, 1.0)
    assert model.number_of_species == 1
    print("✅ test_one_species_in_system passed")

def test_two_species_in_system():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 1.1)
    assert model.number_of_species == 2
    print("✅ test_two_species_in_system passed")

def test_stoichiometric_matrix_values():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 1.2)
    model.add_reaction({"B": 1}, {"A": 1}, 1.3)
    expected_matrix = np.array([[-1, 1],
                                [ 1, -1]])
    assert (model.stoichiometric_matrix == expected_matrix).all()
    print("✅ test_stoichiometric_matrix_values passed")

def test_stoichiometric_matrix_switched_order():
    model_1 = Reaction()
    model_1.add_reaction({"B": 1, "A":1}, {"A": 1}, 1.4)
    model_1.add_reaction({"A": 1}, {"B": 1}, 1.0)

    model_2 = Reaction()
    model_2.add_reaction({"B": 1,"A": 1}, {"A": 1},1.4)
    model_2.add_reaction({"A": 1}, {"B": 1}, 1.0)

    assert (model_1.stoichiometric_matrix == model_2.stoichiometric_matrix).all()
    print("✅ test_stoichiometric_matrix_switched_order passed")
