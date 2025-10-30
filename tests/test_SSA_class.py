from stochastic_framework import Reaction, SSA
import numpy as np


def test_compartments_correct():
    model = Reaction()
    model.add_reaction({"A": 1, "B": 1}, {"A": 1}, 0.1)
    
    ssa_simulator = SSA(model)
    ssa_simulator.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=50.0,
        initial_conditions=np.array([[10, 5],
                                        [20, 15]]),
        timestep=0.5,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )


    assert ssa_simulator.n_compartments == 2


def test_zero_compartment_instance():

    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.2)
    
    ssa_simulator = SSA(model)
    try:
        ssa_simulator.set_conditions(
            n_compartments=0,
            domain_length=10.0,
            total_time=50.0,
            initial_conditions=np.array([]),
            timestep=0.5,
            Macroscopic_diffusion_rates=[0.01, 0.02]
        )
    except ValueError as e:
        print(e)

def test_wrong_init_to_compartment_1():
    """
    Testing whether having just one initial condition for one molecule results in an assertion error.
    """
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.3)
    
    ssa_simulator = SSA(model)
    try:
        ssa_simulator.set_conditions(
            n_compartments=3,
            domain_length=10.0,
            total_time=50.0,
            initial_conditions=np.array([10,5]),
            timestep=0.5,
            Macroscopic_diffusion_rates=[0.01, 0.02]
        )
    except ValueError as e:
        print(e)

def test_wrong_init_to_compartment_2():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.3)
    
    ssa_simulator = SSA(model)
    try:
        ssa_simulator.set_conditions(
            n_compartments=3,
            domain_length=10.0,
            total_time=50.0,
            initial_conditions=np.array([[10.0],
                                        [10.0]]),
            timestep=0.5,
            Macroscopic_diffusion_rates=[0.01, 0.02]
        )
    except ValueError as e:
        print(e)

        
def test_wrong_init_to_compartment_2():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.3)
   
    
    ssa_simulator = SSA(model)
    try:
        ssa_simulator.set_conditions(
            n_compartments=3,
            domain_length=10.0,
            total_time=50.0,
            initial_conditions=np.array([[10.0,10.0,10.0],
                                        [10.0,10.0,10.0]]),
            timestep=0.5,
            Macroscopic_diffusion_rates=[0.01, 0.02]
        )
    except ValueError as e:
        print(e)

def test_correct_initial_conditions():

    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.3)
   
    
    ssa_simulator = SSA(model)
    
    ssa_simulator.set_conditions(
        n_compartments=3,
        domain_length=10.0,
        total_time=50.0,
        initial_conditions=np.array([[10.0,10.0, 10.0],
                                    [10.0,10.0, 10.0]]),
        timestep=0.5,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )
    
    assert (ssa_simulator.initial_conditions.shape == (2,3))

def test_negative_initial_conditions():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.3)
   
    
    ssa_simulator = SSA(model)
    try:
        ssa_simulator.set_conditions(
            n_compartments=2,
            domain_length=10.0,
            total_time=50.0,
            initial_conditions=np.array([[10.0, -5.0],
                                        [15.0, 10.0]]),
            timestep=0.5,
            Macroscopic_diffusion_rates=[0.01, 0.02]
        )
    except ValueError as e:
        print(e)


def test_tensor_shape():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.1)
    ssa = SSA(model)
    
    init = np.array([[10, 5], [20, 15]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )
    
    tensor = ssa._generate_dataframes()
    expected_shape = (len(ssa.timevector), len(ssa.species_list),ssa.n_compartments)
    assert tensor.shape == expected_shape

def test_initial_conds_is_in_tensor():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.1)
    ssa = SSA(model)
    
    init = np.array([[10, 5], [20, 15]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )
    
    tensor = ssa._generate_dataframes()
    assert (tensor[0,:,:] == init).all()


def test_jump_rates_are_correct():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.1)
    ssa = SSA(model)
    
    init = np.array([[10, 5], [20, 15]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.02, 0.04]
    )
    
    expected_jump_rates = [0.02 / (ssa.h ** 2), 0.04 / (ssa.h ** 2)]
    assert ssa.jump_rate_list == expected_jump_rates


def test_raises_error_with_wrong_propensity_dim():

    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.2)
    ssa = SSA(model)

    init = np.array([[10, 5], [20, 15]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )

    wrong_propensity_vector = np.zeros(5)  # Incorrect size

    try:
        ssa._propensity_calculation(
            dataframe=ssa.initial_conditions,
            propensity_vector=wrong_propensity_vector
        )
    except AssertionError as e:
        print(e)


def test_zero_order_propensity_diffusion():

    model = Reaction()
    model.add_reaction({}, {"A": 1}, 1.0)  # Zero-order reaction
    ssa = SSA(model)

    init = np.array([[10,1]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01]
    )

    propensity_vector = np.zeros(ssa.n_compartments*len(ssa.species_list) + ssa.n_compartments*ssa.reaction_system.number_of_reactions)

    propensity_output = ssa._propensity_calculation(
        dataframe=ssa.initial_conditions,
        propensity_vector=propensity_vector
    )
    # For diffusion of species A, propensity should equal jump rate * number of A in each compartment
    jump_rate = ssa.jump_rate_list[0]
    expected_propensity_compartment_0 = jump_rate * init[0,0]*2.0
    expected_propensity_compartment_1 = jump_rate * init[0,1]*2.0

    assert propensity_output[0] == expected_propensity_compartment_0
    assert propensity_output[1] == expected_propensity_compartment_1

def test_zero_order_propensity():

    model = Reaction()
    model.add_reaction({}, {"A": 1}, 1.0)  # Zero-order reaction
    ssa = SSA(model)

    init = np.array([[10,1]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01]
    )

    propensity_vector = np.zeros(ssa.n_compartments*len(ssa.species_list) + ssa.n_compartments*ssa.reaction_system.number_of_reactions)

    propensity_output = ssa._propensity_calculation(
        dataframe=ssa.initial_conditions,
        propensity_vector=propensity_vector
    )
    # For zero-order reaction, propensity should equal the rate constant in each compartment
    h = ssa.domain_length / ssa.n_compartments
    expected_propensity = 1.0*h
    for i in range(ssa.n_compartments):
        assert propensity_output[ssa.n_compartments*len(ssa.species_list) + i] == expected_propensity


def test_first_order_propensity():

    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.5)  # First-order reaction
    ssa = SSA(model)

    init = np.array([[10, 5], [0, 0]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01, 0.1]
    )

    propensity_vector = np.zeros(ssa.n_compartments*len(ssa.species_list) + ssa.n_compartments*ssa.reaction_system.number_of_reactions)

    propensity_output = ssa._propensity_calculation(
        dataframe=ssa.initial_conditions,
        propensity_vector=propensity_vector
    )
    # For first-order reaction, propensity should equal rate constant * number of A in each compartment
    rate = 0.5
    for i in range(ssa.n_compartments):
        expected_propensity = rate * init[0,i]
        assert propensity_output[ssa.n_compartments*len(ssa.species_list) + i] == expected_propensity


def test_second_order_propensity():

    model = Reaction()
    model.add_reaction({"A": 2}, {"B": 1}, 0.2)  # Second-order reaction
    ssa = SSA(model)

    init = np.array([[10, 5], [0, 0]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01, 0.1]
    )

    propensity_vector = np.zeros(ssa.n_compartments*len(ssa.species_list) + ssa.n_compartments*ssa.reaction_system.number_of_reactions)

    propensity_output = ssa._propensity_calculation(
        dataframe=ssa.initial_conditions,
        propensity_vector=propensity_vector
    )
    # For second-order reaction A + A, propensity should equal rate constant * n_A * (n_A - 1) / h in each compartment
    rate = 0.2
    h = ssa.domain_length / ssa.n_compartments
    for i in range(ssa.n_compartments):
        n_A = init[0,i]
        expected_propensity = rate * n_A * (n_A - 1) / h
        assert propensity_output[ssa.n_compartments*len(ssa.species_list) + i] == expected_propensity

def test_zero_and_first():

    model = Reaction()
    model.add_reaction({}, {"A": 1}, 1.0)  # Zero-order reaction
    model.add_reaction({"A": 1}, {"B": 1}, 0.5)  # First-order reaction
    ssa = SSA(model)

    init = np.array([[10, 5], [0, 0]])
    ssa.set_conditions(
        n_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01, 0.1]
    )

    propensity_vector = np.zeros(ssa.n_compartments*len(ssa.species_list) + ssa.n_compartments*ssa.reaction_system.number_of_reactions)

    propensity_output = ssa._propensity_calculation(
        dataframe=ssa.initial_conditions,
        propensity_vector=propensity_vector
    )
    # Check zero-order reaction propensities
    h = ssa.domain_length / ssa.n_compartments
    expected_zero_order_propensity = 1.0*h
    for i in range(ssa.n_compartments):
        assert propensity_output[ssa.n_compartments*len(ssa.species_list) + i] == expected_zero_order_propensity

    # Check first-order reaction propensities
    rate = 0.5
    for i in range(ssa.n_compartments):
        expected_first_order_propensity = rate * init[0,i]
        assert propensity_output[ssa.n_compartments*len(ssa.species_list) + ssa.n_compartments + i] == expected_first_order_propensity


def test_multiple_compartments():

    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.3)  # First-order reaction
    model.add_reaction({"A":1, "B":1}, {}, 0.4)  # Second-order reaction
    ssa = SSA(model)

    init = np.array([[10, 5, 15, 10, 5, 3], [0, 0, 0, 1,2,3]])
    ssa.set_conditions(
        n_compartments=6,
        domain_length=30.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )

    propensity_vector = np.zeros(ssa.n_compartments*len(ssa.species_list) + ssa.n_compartments*ssa.reaction_system.number_of_reactions)
    propensity_output = ssa._propensity_calculation(
        dataframe=ssa.initial_conditions,
        propensity_vector=propensity_vector
    )
    # Check first the diffusion is correct
    jump_rate_A = ssa.jump_rate_list[0]
    jump_rate_B = ssa.jump_rate_list[1]
    for i in range(ssa.n_compartments):
        expected_diffusion_A = jump_rate_A * init[0,i]*2.0
        expected_diffusion_B = jump_rate_B * init[1,i]*2.0
        assert propensity_output[i] == expected_diffusion_A
        assert propensity_output[ssa.n_compartments + i] == expected_diffusion_B
    
    rate_first = 0.3
    for i in range(ssa.n_compartments):
        expected_first_order_propensity = rate_first * init[0,i]
        assert propensity_output[ssa.n_compartments*len(ssa.species_list) + i] == expected_first_order_propensity

    rate_second = 0.4
    h = ssa.domain_length / ssa.n_compartments
    for i in range(ssa.n_compartments):
        n_A = init[0,i]
        n_B = init[1,i]
        expected_second_order_propensity = rate_second * n_A * n_B / h
        assert propensity_output[ssa.n_compartments*len(ssa.species_list) + ssa.n_compartments + i] == expected_second_order_propensity