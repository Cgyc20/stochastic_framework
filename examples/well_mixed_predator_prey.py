import numpy as np
from stochastic_framework import Reaction
from stochastic_framework.well_mixed_stochastic_sim import WellMixedSSA
import matplotlib.pyplot as plt


def analytic_lotka_volterra(t, A0, B0, k1, k2, k3, volume):
    """
    Analytic solution for the Lotka-Volterra predator-prey system.
    
    For the system:
    A + B -> 2B  (predation, rate k1)
    B -> 0       (predator death, rate k2)
    0 -> A       (prey birth, rate k3)
    
    This system has oscillatory dynamics. For small fluctuations around equilibrium,
    we can compute the equilibrium point:
    A_eq = k2 * V / k1
    B_eq = k3 * V / k1
    
    The system oscillates around these values with period ~ 2π/sqrt(k1*k2*k3/V)
    
    Note: Full analytic solution requires elliptic integrals. Here we return
    the equilibrium values for comparison.
    
    Parameters
    ----------
    t : array
        Time points
    A0, B0 : int
        Initial molecule counts
    k1, k2, k3 : float
        Rate constants
    volume : float
        System volume
    
    Returns
    -------
    tuple
        (A_equilibrium, B_equilibrium) - equilibrium values
    """
    A_eq = (k2 * volume) / k1
    B_eq = (k3 * volume) / k1
    return A_eq, B_eq


def main():
    # -----------------------
    # 1. Define Reaction System
    # -----------------------
    ReactionData = Reaction()

    # Lotka-Volterra predator-prey system:
    # 0 -> A        (prey birth)
    # A + B -> 2B   (predation)
    # B -> 0        (predator death)

    k_prey_birth = 1.0
    k_predation = 0.01      # Changed from 0.0001 to 0.01
    k_pred_death = 0.05
    A_initial = 50         # Changed from 200
    B_initial = 1          # Changed from 80

    # Add reactions
    ReactionData.add_reaction({}, {"A": 1}, k_prey_birth)           # Prey birth
    ReactionData.add_reaction({"A": 1, "B": 1}, {"B": 2}, k_predation)  # Predation
    ReactionData.add_reaction({"B": 1}, {}, k_pred_death)           # Predator death

    # Show what we've got
    print("\n--- Lotka-Volterra Predator-Prey System ---")
    ReactionData.print_reactions()
    print(ReactionData.show_stoichiometry())

    # -----------------------
    # 2. Define Well-Mixed SSA System
    # -----------------------
    WellMixed = WellMixedSSA(ReactionData)

    # Initial conditions (start away from equilibrium to see oscillations)

    initial_conditions = np.array([A_initial, B_initial], dtype=int)

    volume = 1.0  # Volume of the well-mixed system

    # Calculate equilibrium points
    A_eq, B_eq = analytic_lotka_volterra(None, A_initial, B_initial, 
                                         k_predation, k_pred_death, k_prey_birth, volume)
    print(f"\n📊 Equilibrium values:")
    print(f"   Prey (A) equilibrium: {A_eq:.2f} molecules")
    print(f"   Predator (B) equilibrium: {B_eq:.2f} molecules")

    # -----------------------
    # 3. Set Simulation Conditions
    # -----------------------
    WellMixed.set_conditions(
        volume=volume,
        total_time=1000.0,  # Longer time to see multiple oscillations
        initial_conditions=initial_conditions,
        timestep=0.2
    )

    # -----------------------
    # 4. Run Simulation
    # -----------------------
    print("\nRunning well-mixed SSA simulation...")
    average_output = WellMixed.run_simulation(n_repeats=1000)  # Fewer repeats for faster execution

    # -----------------------
    # 5. Save Results
    # -----------------------
    save_path = "/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/well_mixed_predator_prey_data.npz"

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
             label='Prey (A)', linewidth=2.5, color='#2E86AB', alpha=0.8)
    ax1.plot(WellMixed.timevector, average_output[:, 1], 
             label='Predator (B)', linewidth=2.5, color='#C73E1D', alpha=0.8)
    ax1.axhline(y=A_eq, color='#2E86AB', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axhline(y=B_eq, color='#C73E1D', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Number of Molecules', fontsize=12)
    ax1.set_title('Population Dynamics Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Middle plot: Phase space
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(average_output[:, 0], average_output[:, 1], 
             linewidth=2, color='#6A4C93', alpha=0.7)
    ax2.scatter(average_output[0, 0], average_output[0, 1], 
                s=100, color='green', marker='o', zorder=5, label='Start')
    ax2.scatter(average_output[-1, 0], average_output[-1, 1], 
                s=100, color='red', marker='s', zorder=5, label='End')
    ax2.scatter(A_eq, B_eq, s=150, color='gold', marker='*', 
                zorder=5, edgecolors='black', linewidths=1.5, label='Equilibrium')
    ax2.set_xlabel('Prey (A)', fontsize=12)
    ax2.set_ylabel('Predator (B)', fontsize=12)
    ax2.set_title('Phase Space Trajectory', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Right plot: Ratio dynamics
    ax3 = plt.subplot(1, 3, 3)
    ratio = average_output[:, 0] / np.maximum(average_output[:, 1], 1.0)
    ax3.plot(WellMixed.timevector, ratio, linewidth=2, color='#F18F01')
    ax3.axhline(y=A_eq/B_eq, color='black', linestyle='--', linewidth=1.5, 
                label=f'Eq. Ratio = {A_eq/B_eq:.2f}')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Prey/Predator Ratio', fontsize=12)
    ax3.set_title('Population Ratio Dynamics', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/Users/charliecameron/CodingHub/PhD/Y2_code/stochastic_framework/data/well_mixed_predator_prey_plot.png', dpi=150)
    print("\n📊 Plot saved to: data/well_mixed_predator_prey_plot.png")
    
    # Calculate and print statistics
    mean_prey = np.mean(average_output[len(average_output)//2:, 0])
    mean_predator = np.mean(average_output[len(average_output)//2:, 1])
    
    print(f"\n📈 Statistics (second half of simulation):")
    print(f"   Mean prey population: {mean_prey:.2f} (equilibrium: {A_eq:.2f})")
    print(f"   Mean predator population: {mean_predator:.2f} (equilibrium: {B_eq:.2f})")
    print(f"   Mean prey/predator ratio: {mean_prey/mean_predator:.2f} (equilibrium: {A_eq/B_eq:.2f})")
    
    plt.show()


if __name__ == "__main__":
    main()
