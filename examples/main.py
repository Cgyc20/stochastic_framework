from stochastic_framework import Reaction, SSA
import numpy as np


def main():
    ReactionData = Reaction()


    r_11 = 1.0
    r_12 = 2.0
    r_2 = 1.0
    r_3 = 1.0

    Du = 0.0005
    Dv = 25*Du
    omega = 500
    domain_length = 10.0
    n_compartments = 10

    rescaled_r11 = r_11/omega
    rescaled_r12 = r_12/omega
    rescaled_r2 = r_2/omega
    rescaled_r3 = r_3

    rescaled_Du = Du * domain_length**2
    rescaled_Dv = Dv * domain_length**2


    U_steady_state_conc = r_11*r_3/(r_12*r_2)
    V_steady_state_conc = (r_11**2)*r_3/(r_12*r_2**2)


    rescaled_U_steady_state_conc = U_steady_state_conc*omega
    rescaled_V_steady_state_conc = V_steady_state_conc*omega

    print(U_steady_state_conc, V_steady_state_conc)
    print(rescaled_U_steady_state_conc, rescaled_V_steady_state_conc)

    h = domain_length / n_compartments  # compartment size
    U_steady_state_mass = rescaled_U_steady_state_conc * h  # mass per compartment (may be fractional)
    V_steady_state_mass = rescaled_V_steady_state_conc * h

    U_initial = np.ones(n_compartments, dtype = int)*int(round(U_steady_state_mass))
    V_initial = np.ones(n_compartments, dtype = int)*int(round(V_steady_state_mass))





    ReactionData.add_reaction({"U": 2}, {"U": 3}, rescaled_r11)
    ReactionData.add_reaction({"U": 2}, {"U": 2, "V": 1}, rescaled_r12)
    ReactionData.add_reaction({"U":1, "V":1},{"V":1}, rescaled_r2)
    ReactionData.add_reaction({"V":1},{}, rescaled_r3)


    ReactionData.print_reactions()
    stoich_df = ReactionData.show_stoichiometry()
    print(stoich_df)

    SSA_class = SSA(ReactionData)

    print(SSA_class.stoichiometric_matrix)

    initial_conditions = np.zeros((2, n_compartments), dtype = int)

    initial_conditions[0,:] = U_initial
    initial_conditions[1,:] = V_initial



    SSA_class.set_conditions(
        n_compartments=n_compartments,
        domain_length=domain_length,
        total_time=40.0,
        initial_conditions=initial_conditions,
        timestep=0.1,
        Macroscopic_diffusion_rates=[rescaled_Du, rescaled_Dv]
    )

    SSA_class._propensity_calculation(
        dataframe=SSA_class.initial_conditions,
        propensity_vector=np.zeros(SSA_class.n_compartments*len(SSA_class.species_list) + SSA_class.n_compartments*SSA_class.reaction_system.number_of_reactions)
    )

    average_output = SSA_class.run_simulation(n_repeats=5)

    SSA_class.save_simulation_data(
        filename="/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/SSA_data.npz",
        simulation_result=average_output
    ) 

if __name__ == "__main__":
    main()