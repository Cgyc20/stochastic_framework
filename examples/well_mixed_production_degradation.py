import numpy as np
from stochastic_framework import Reaction
from stochastic_framework.well_mixed_stochastic_sim import WellMixedSSA
import matplotlib.pyplot as plt


def analytic_solution(t, k_prod, k_deg, volume):
    """
    Analytic solution for the mean of a simple production-degradation system.
    
    For the system:
    0 -> A with rate k_prod (constant production)
    A -> 0 with rate k_deg (linear degradation)
    
    The mean follows: dE[A]/dt = k_prod * V - k_deg * E[A]
    Steady state: E[A]_ss = (k_prod * V) / k_deg
    
    Solution: E[A](t) = E[A]_ss * (1 - exp(-k_deg * t))
    (assuming A(0) = 0)
    
    Parameters
    ----------
    t : array
        Time points
    k_prod : float
        Production rate constant
    k_deg : float
        Degradation rate constant
    volume : float
        System volume
    
    Returns
    -------
    array
        Expected number of molecules at each time point
    """
    steady_state = (k_prod * volume) / k_deg
    return steady_state * (1 - np.exp(-k_deg * t))


def main():
    # -----------------------
    # 1. Define Reaction System
    # -----------------------
    ReactionData = Reaction()

    # Simple production-degradation system:
    # 0 -> A   (constant production)
    # A -> 0   (degradation)

    k_prod = 5.0   # production rate
    k_deg = 0.5    # degradation rate

    # Add reactions
    ReactionData.add_reaction({}, {"A": 1}, k_prod)      # Production
    ReactionData.add_reaction({"A": 1}, {}, k_deg)       # Degradation

    # Show what we've got
    print("\n--- Reaction System ---")
    ReactionData.print_reactions()
    print(ReactionData.show_stoichiometry())

    # -----------------------
    # 2. Define Well-Mixed SSA System
    # -----------------------
    WellMixed = WellMixedSSA(ReactionData)

    # Initial conditions (start with zero molecules)
    A_initial = np.array([0], dtype=int)

    volume = 1.0  # Volume of the well-mixed system

    # Calculate expected steady state
    steady_state = (k_prod * volume) / k_deg
    print(f"\n📊 Expected steady state: {steady_state:.2f} molecules")

    # -----------------------
    # 3. Set Simulation Conditions
    # -----------------------
    WellMixed.set_conditions(
        volume=volume,
        total_time=20.0,
        initial_conditions=A_initial,
        timestep=0.05
    )

    # -----------------------
    # 4. Run Simulation
    # -----------------------
    print("\nRunning well-mixed SSA simulation...")
    average_output = WellMixed.run_simulation(n_repeats=200)

    # -----------------------
    # 5. Save Results
    # -----------------------
    save_path = "/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/well_mixed_prod_deg_data.npz"

    WellMixed.save_simulation_data(
        filename=save_path,
        simulation_result=average_output
    )

    print(f"\n✅ Simulation complete. Data saved to:\n{save_path}")

    # -----------------------
    # 6. Calculate Analytic Solution
    # -----------------------
    analytic_mean = analytic_solution(WellMixed.timevector, k_prod, k_deg, volume)
    
    # -----------------------
    # 7. Plot Results with Comparison
    # -----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Full trajectory
    ax1.plot(WellMixed.timevector, average_output[:, 0], 
             label='Stochastic SSA (averaged)', linewidth=2.5, color='#2E86AB', alpha=0.8)
    ax1.plot(WellMixed.timevector, analytic_mean, 
             label='Analytic Solution', linewidth=2, linestyle='--', color='#A23B72')
    ax1.axhline(y=steady_state, color='#F18F01', linestyle=':', linewidth=2, 
                label=f'Steady State = {steady_state:.1f}')
    ax1.set_xlabel('Time', fontsize=13)
    ax1.set_ylabel('Number of Molecules', fontsize=13)
    ax1.set_title('Production-Degradation: Approach to Steady State', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Error over time
    error = np.abs(average_output[:, 0] - analytic_mean)
    relative_error = error / np.maximum(analytic_mean, 1.0) * 100
    
    ax2.plot(WellMixed.timevector, relative_error, linewidth=2, color='#C73E1D')
    ax2.set_xlabel('Time', fontsize=13)
    ax2.set_ylabel('Relative Error (%)', fontsize=13)
    ax2.set_title('Stochastic vs Analytic: Relative Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/well_mixed_prod_deg_plot.png', dpi=150)
    print("\n📊 Plot saved to: data/well_mixed_prod_deg_plot.png")
    
    # Calculate and print error metrics
    mean_relative_error = np.mean(relative_error)
    final_stochastic = average_output[-1, 0]
    final_analytic = analytic_mean[-1]
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Mean relative error: {mean_relative_error:.2f}%")
    print(f"   Final stochastic value: {final_stochastic:.2f}")
    print(f"   Final analytic value: {final_analytic:.2f}")
    print(f"   Difference from steady state: {abs(final_stochastic - steady_state):.2f}")
    print(f"   % of steady state reached: {(final_stochastic/steady_state)*100:.1f}%")
    
    plt.show()


if __name__ == "__main__":
    main()
