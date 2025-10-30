import numpy as np
from stochastic_framework import Reaction, SSA

# -----------------------
# 1. Define Reaction System
# -----------------------
ReactionData = Reaction()

# Simple system:
# A -> 2A   (birth)
# A -> 0    (death)
# D_A       (diffusion)

r_birth = 0.5
r_death = 0.8
D_A = 0.001

domain_length = 10.0
n_compartments = 10
omega = 10  # scaling factor for molecular counts

# Rescale for SSA
rescaled_birth = r_birth
rescaled_death = r_death
rescaled_DA = D_A * domain_length**2

# Add reactions
ReactionData.add_reaction({"A": 1}, {"A": 2}, rescaled_birth)
ReactionData.add_reaction({"A": 1}, {}, rescaled_death)

# Show what we’ve got
print("\n--- Reaction System ---")
ReactionData.print_reactions()
print(ReactionData.show_stoichiometry())

# -----------------------
# 2. Define SSA System
# -----------------------
SSA_class = SSA(ReactionData)

# Initial conditions (uniform)
A_initial = np.ones(n_compartments, dtype=int) * 30
initial_conditions = np.zeros((1, n_compartments), dtype=int)
initial_conditions[0, :] = A_initial

# -----------------------
# 3. Set Simulation Conditions
# -----------------------
SSA_class.set_conditions(
    n_compartments=n_compartments,
    domain_length=domain_length,
    total_time=30.0,
    initial_conditions=initial_conditions,
    timestep=0.1,
    Macroscopic_diffusion_rates=[rescaled_DA]
)

# -----------------------
# 4. Run Simulation
# -----------------------
print("\nRunning SSA simulation...")
average_output = SSA_class.run_simulation(n_repeats=50)

# -----------------------
# 5. Save Results
# -----------------------
save_path = "/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/SSA_data.npz"

SSA_class.save_simulation_data(
    filename=save_path,
    simulation_result=average_output
)

print(f"\n✅ Simulation complete. Data saved to:\n{save_path}")
