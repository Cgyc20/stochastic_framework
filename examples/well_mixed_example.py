import numpy as np
from stochastic_framework import Reaction
from stochastic_framework.well_mixed_stochastic_sim import WellMixedSSA
import matplotlib.pyplot as plt


def analytic_solution(t, A0, r_birth, r_death):
    """
    Analytic solution for the mean of the birth-death process.
    
    For the system:
    A -> 2A with rate r_birth (net production rate = r_birth)
    A -> 0 with rate r_death
    
    The mean follows: dE[A]/dt = (r_birth - r_death) * E[A]
    Solution: E[A](t) = A0 * exp((r_birth - r_death) * t)
    
    Parameters
    ----------
    t : array
        Time points
    A0 : int
        Initial number of molecules
    r_birth : float
        Birth rate constant
    r_death : float
        Death rate constant
    
    Returns
    -------
    array
        Expected number of molecules at each time point
    """
    net_rate = r_birth - r_death
    return A0 * np.exp(net_rate * t)


def main():
    # -----------------------
    # 1. Define Reaction System
    # -----------------------
    ReactionData = Reaction()

    # Simple birth-death system in a well-mixed volume:
    # A -> 2A   (birth/autocatalysis)
    # A -> 0    (death/degradation)

    r_birth = 0.5   # birth rate
    r_death = 0.8   # death rate

    # Add reactions
    ReactionData.add_reaction({"A": 1}, {"A": 2}, r_birth)
    ReactionData.add_reaction({"A": 1}, {}, r_death)

    # Show what we've got
    print("\n--- Reaction System ---")
    ReactionData.print_reactions()
    print(ReactionData.show_stoichiometry())

    # -----------------------
    # 2. Define Well-Mixed SSA System
    # -----------------------
    WellMixed = WellMixedSSA(ReactionData)

    # Initial conditions (single well-mixed compartment)
    A_initial = np.array([50], dtype=int)  # Start with 50 molecules of A

    volume = 1.0  # Volume of the well-mixed system

    # -----------------------
    # 3. Set Simulation Conditions
    # -----------------------
    WellMixed.set_conditions(
        volume=volume,
        total_time=30.0,
        initial_conditions=A_initial,
        timestep=0.1
    )

    # -----------------------
    # 4. Run Simulation
    # -----------------------
    print("\nRunning well-mixed SSA simulation...")
    average_output = WellMixed.run_simulation(n_repeats=100)

    # -----------------------
    # 5. Save Results
    # -----------------------
    save_path = "/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/well_mixed_SSA_data.npz"

    WellMixed.save_simulation_data(
        filename=save_path,
        simulation_result=average_output
    )

    print(f"\n✅ Simulation complete. Data saved to:\n{save_path}")

    # -----------------------
    # 6. Calculate Analytic Solution
    # -----------------------
    analytic_mean = analytic_solution(WellMixed.timevector, A_initial[0], r_birth, r_death)
    
    # -----------------------
    # 7. Plot Results with Comparison
    # -----------------------
    plt.figure(figsize=(12, 7))
    
    # Plot stochastic simulation
    plt.plot(WellMixed.timevector, average_output[:, 0], 
             label='Stochastic SSA (averaged)', linewidth=2.5, color='#2E86AB', alpha=0.8)
    
    # Plot analytic solution
    plt.plot(WellMixed.timevector, analytic_mean, 
             label='Analytic Solution', linewidth=2, linestyle='--', color='#A23B72')
    
    plt.xlabel('Time', fontsize=13)
    plt.ylabel('Number of Molecules', fontsize=13)
    plt.title('Well-Mixed Birth-Death Process: Stochastic vs Analytic', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/well_mixed_plot.png', dpi=150)
    print("\n📊 Plot saved to: data/well_mixed_plot.png")
    
    # Calculate and print error metrics
    relative_error = np.mean(np.abs(average_output[:, 0] - analytic_mean) / analytic_mean) * 100
    print(f"\n📈 Mean relative error: {relative_error:.2f}%")
    print(f"   Final stochastic value: {average_output[-1, 0]:.2f}")
    print(f"   Final analytic value: {analytic_mean[-1]:.2f}")
    
    plt.show()


if __name__ == "__main__":
    main()
