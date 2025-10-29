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