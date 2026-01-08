import numpy as np
from stochastic_framework import Reaction
from stochastic_framework.well_mixed_stochastic_sim import WellMixedSSA
import matplotlib.pyplot as plt


def main():
    # -----------------------
    # 1. Define Reaction System
    # -----------------------
    ReactionData = Reaction()

    # Schnakenberg/Turing-like autocatalytic system:
    # 2U -> 3U         (U self-activation)
    # 2U -> 2U + V     (V produced by U)
    # U + V -> V       (U inhibits itself via V)
    # V -> 0           (V decay)

    # Parameters
    bar_r_11 = 1.0      # U self-activation
    bar_r_12 = 1.0      # V production by U
    bar_r_2 = 2.0       # U-V interaction (inhibition)
    bar_r_3 = 0.6       # V decay
    
    omega = 100        # scaling factor for molecule counts
    domain_length = 1.0

    h = 1.0
    # Rescale rates for stochastic simulation
    r_11 = bar_r_11 / omega
    r_12 = bar_r_12 / omega
    r_2 = bar_r_2 / omega
    r_3 = bar_r_3

    # Calculate steady state concentrations
    dimensionless_U_ss_conc = bar_r_11 * bar_r_3 / (bar_r_12 * bar_r_2)
    dimensionless_V_ss_conc = (bar_r_11**2) * bar_r_3 / (bar_r_12 * bar_r_2**2)

    U_steady_state_conc = dimensionless_U_ss_conc * omega
    V_steady_state_conc = dimensionless_V_ss_conc * omega
    U_steady_state_mass = U_steady_state_conc * h
    V_steady_state_mass = V_steady_state_conc * h

    print(f"\n--- System Parameters ---")
    print(f"Omega (scaling): {omega}")
    print(f"U steady state mass: {U_steady_state_mass:.2f}")
    print(f"V steady state mass: {V_steady_state_mass:.2f}")

    # Add reactions
    ReactionData.add_reaction({"U": 2}, {"U": 3}, r_11)
    ReactionData.add_reaction({"U": 2}, {"U": 2, "V": 1}, r_12)
    ReactionData.add_reaction({"U": 1, "V": 1}, {"V": 1}, r_2)
    ReactionData.add_reaction({"V": 1}, {}, r_3)

    # Show what we've got
    print("\n--- Reaction System ---")
    ReactionData.print_reactions()
    print(ReactionData.show_stoichiometry())

    # -----------------------
    # 2. Define Well-Mixed SSA System
    # -----------------------
    WellMixed = WellMixedSSA(ReactionData)

    # Initial conditions (start at steady state)
    U_initial = round(U_steady_state_mass)
    V_initial = round(V_steady_state_mass)
    initial_grid = np.array([U_initial, V_initial], dtype=int)

    print(f"\n--- Initial Conditions ---")
    print(f"U_initial: {U_initial}")
    print(f"V_initial: {V_initial}")

    volume = h  # Volume matches compartment size

    # -----------------------
    # 3. Set Simulation Conditions
    # -----------------------
    WellMixed.set_conditions(
        volume=volume,
        total_time=30.0,
        initial_conditions=initial_grid,
        timestep=0.1
    )

    # -----------------------
    # 4. Run Simulation
    # -----------------------
    print("\nRunning well-mixed SSA simulation...")
    average_output = WellMixed.run_simulation(n_repeats=50)

    # -----------------------
    # 5. Save Results
    # -----------------------
    save_path = "/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/well_mixed_autocatalytic_data.npz"

    WellMixed.save_simulation_data(
        filename=save_path,
        simulation_result=average_output
    )

    print(f"\n✅ Simulation complete. Data saved to:\n{save_path}")

    # -----------------------
    # 6. Plot Results
    # -----------------------
    fig = plt.figure(figsize=(16, 5))
    
    # Left plot: Time series
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(WellMixed.timevector, average_output[:, 0], 
             label='U (Activator)', linewidth=2.5, color='#2E86AB', alpha=0.8)
    ax1.plot(WellMixed.timevector, average_output[:, 1], 
             label='V (Inhibitor)', linewidth=2.5, color='#C73E1D', alpha=0.8)
    ax1.axhline(y=U_initial, color='#2E86AB', linestyle=':', linewidth=1.5, alpha=0.5, label='U steady state')
    ax1.axhline(y=V_initial, color='#C73E1D', linestyle=':', linewidth=1.5, alpha=0.5, label='V steady state')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Number of Molecules', fontsize=12)
    ax1.set_title('Autocatalytic System: Time Series', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Middle plot: Phase space
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(average_output[:, 0], average_output[:, 1], 
             linewidth=2, color='#6A4C93', alpha=0.7)
    ax2.scatter(average_output[0, 0], average_output[0, 1], 
                s=100, color='green', marker='o', zorder=5, label='Start')
    ax2.scatter(average_output[-1, 0], average_output[-1, 1], 
                s=100, color='red', marker='s', zorder=5, label='End')
    ax2.scatter(U_initial, V_initial, s=150, color='gold', marker='*', 
                zorder=5, edgecolors='black', linewidths=1.5, label='Steady State')
    ax2.set_xlabel('U (Activator)', fontsize=12)
    ax2.set_ylabel('V (Inhibitor)', fontsize=12)
    ax2.set_title('Phase Space Trajectory', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Right plot: U/V Ratio
    ax3 = plt.subplot(1, 3, 3)
    ratio = average_output[:, 0] / np.maximum(average_output[:, 1], 1.0)
    expected_ratio = U_initial / V_initial
    ax3.plot(WellMixed.timevector, ratio, linewidth=2, color='#F18F01')
    ax3.axhline(y=expected_ratio, color='black', linestyle='--', linewidth=1.5, 
                label=f'Expected Ratio = {expected_ratio:.2f}')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('U/V Ratio', fontsize=12)
    ax3.set_title('Activator/Inhibitor Ratio', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/well_mixed_autocatalytic_plot.png', dpi=150)
    print("\n📊 Plot saved to: data/well_mixed_autocatalytic_plot.png")
    
    # Calculate and print statistics
    mean_U = np.mean(average_output[len(average_output)//2:, 0])
    mean_V = np.mean(average_output[len(average_output)//2:, 1])
    
    print(f"\n📈 Statistics (second half of simulation):")
    print(f"   Mean U population: {mean_U:.2f} (expected: {U_initial:.2f})")
    print(f"   Mean V population: {mean_V:.2f} (expected: {V_initial:.2f})")
    print(f"   Mean U/V ratio: {mean_U/mean_V:.2f} (expected: {expected_ratio:.2f})")
    print(f"   U deviation: {abs(mean_U - U_initial)/U_initial * 100:.1f}%")
    print(f"   V deviation: {abs(mean_V - V_initial)/V_initial * 100:.1f}%")
    
    plt.show()


if __name__ == "__main__":
    main()
