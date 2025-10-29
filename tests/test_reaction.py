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

