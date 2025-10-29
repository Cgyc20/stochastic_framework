from stochastic_framework import Reaction

def test_valid_two_species_reaction():
    model = Reaction()
    model.add_reaction({"A": 1, "B": 1}, {"A": 1}, 0.1)
    assert len(model.reaction_set) == 1
    print("✅ test_valid_two_species_reaction passed")

def test_raises_value_error_more_than_two():
    model = Reaction()
    try:
        model.add_reaction({"A": 1, "B": 1}, {"C": 1}, 0.2)
    except ValueError:
        print("✅ test_raises_value_error_more_than_two passed")
        return
    # If we get here → no error was raised → fail the test
    print("❌ test_raises_value_error_more_than_two failed")


def test_raises_error_no_products_reactants():
    model = Reaction()
    try:
        model.add_reaction({}, {}, 0.3)
    except ValueError:
        print("✅ test_raises_error_no_products_reactants passed")
        return
    # If we get here → no error was raised → fail the test
    print("❌ test_raises_error_no_products_reactants failed")


def test_error_if_more_than_two_reacting_molecules():
    model = Reaction()
    try:
        model.add_reaction({"A": 3}, {"B": 1}, 0.4)
    except ValueError:
        print("✅ test_error_if_more_than_two_reacting_molecules passed")
        return
    # If we get here → no error was raised → fail the test
    print("❌ test_error_if_more_than_two_reacting_molecules failed")


def test_correct_reaction_type_zeroth_order():
    model = Reaction()
    model.add_reaction({}, {"A": 1}, 0.2)

    dictionary_output = model.reaction_set[0]
    reaction_type = dictionary_output["reaction_type"]

    assert reaction_type == 'zero_order'
    print("✅ test_correct_reaction_type_zeroth_order passed")


def test_correct_reaction_type_first_order():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.5)

    dictionary_output = model.reaction_set[0]
    reaction_type = dictionary_output["reaction_type"]

    assert reaction_type == 'first_order'
    print("✅ test_correct_reaction_type passed")

def test_correct_reaction_type_second_order():
    model = Reaction()
    model.add_reaction({"A": 1, "B": 1}, {"A": 1}, 0.6)

    dictionary_output = model.reaction_set[0]
    reaction_type = dictionary_output["reaction_type"]

    assert reaction_type == 'second_order'
    print("✅ test_correct_reaction_type_second_order passed")

def test_no_reactions_added_stoichiometry_error():
    model = Reaction()
    try:
        model.calculate_stoichiometry()
    except ValueError:
        print("✅ test_no_reactions_added_stoichiometry_error passed")
        return
    # If we get here → no error was raised → fail the test
    print("❌ test_no_reactions_added_stoichiometry_error failed")

def test_one_reaction_is_verified():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.7)
    model.calculate_stoichiometry()
    assert model.number_of_reactions == 1
    print("✅ test_one_reaction_is_verified passed")

def test_two_reactions_verified():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.8)
    model.add_reaction({"B": 1}, {"A": 1}, 0.9)
    model.calculate_stoichiometry()
    assert model.number_of_reactions == 2
    print("✅ test_two_reactions_verified passed")

def test_one_species_in_system():
    model = Reaction()
    model.add_reaction({"A": 1}, {"A": 2}, 1.0)
    model.calculate_stoichiometry()
    assert model.number_of_species == 1
    print("✅ test_one_species_in_system passed")

def test_two_species_in_system():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 1.1)
    model.calculate_stoichiometry()
    assert model.number_of_species == 2
    print("✅ test_two_species_in_system passed")

    