from stochastic_framework import Reaction, SSA
import numpy as np


def test_compartments_correct():
    model = Reaction()
    model.add_reaction({"A": 1, "B": 1}, {"A": 1}, 0.1)
    
    ssa_simulator = SSA(model)
    ssa_simulator.set_conditions(
        number_of_compartments=2,
        domain_length=10.0,
        total_time=50.0,
        initial_conditions=np.array([[10, 5],
                                        [20, 15]]),
        timestep=0.5,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )


    assert ssa_simulator.number_of_compartments == 2


def test_zero_compartment_instance():

    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.2)
    
    ssa_simulator = SSA(model)
    try:
        ssa_simulator.set_conditions(
            number_of_compartments=0,
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
            number_of_compartments=3,
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
            number_of_compartments=3,
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
            number_of_compartments=3,
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
        number_of_compartments=3,
        domain_length=10.0,
        total_time=50.0,
        initial_conditions=np.array([[10.0,10.0],
                                    [10.0,10.0],
                                    [10.0,10.0]]),
        timestep=0.5,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )
    
    assert (ssa_simulator.initial_conditions.shape == (3,2))

def test_negative_initial_conditions():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.3)
   
    
    ssa_simulator = SSA(model)
    try:
        ssa_simulator.set_conditions(
            number_of_compartments=2,
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
        number_of_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.01, 0.02]
    )
    
    tensor = ssa._generate_dataframes()
    expected_shape = (len(ssa.timevector), ssa.number_of_compartments, len(ssa.species_list))
    assert tensor.shape == expected_shape

def test_initial_conds_is_in_tensor():
    model = Reaction()
    model.add_reaction({"A": 1}, {"B": 1}, 0.1)
    ssa = SSA(model)
    
    init = np.array([[10, 5], [20, 15]])
    ssa.set_conditions(
        number_of_compartments=2,
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
        number_of_compartments=2,
        domain_length=10.0,
        total_time=5.0,
        initial_conditions=init,
        timestep=1.0,
        Macroscopic_diffusion_rates=[0.02, 0.04]
    )
    
    expected_jump_rates = [0.02 / (ssa.h ** 2), 0.04 / (ssa.h ** 2)]
    assert ssa.jump_rate_list == expected_jump_rates