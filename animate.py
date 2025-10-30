import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

ssa_path = "SSA_data.npz"

def main(ssa_path):
    sns.set_theme(style="whitegrid")

    # Load SSA data
    data = np.load(ssa_path, allow_pickle=True)
    simulation_result = data["simulation_result"]  # shape: (time, species, compartments)
    time_vector = data["timevector"]
    space = data["space"]
    domain_length = data["domain_length"]
    total_time = data["total_time"]
    timestep = data["timestep"]
    h = data["h"]
    n_species = data["n_species"]
    n_compartments = data["n_compartments"]
    reaction_data = data["reaction_data"].item()

    print(f"Simulation Result Shape: {simulation_result.shape}")
    print(f"Time Vector Shape: {time_vector.shape}")
    print(f"Space Vector Shape: {space.shape}")
    print(f"Number of Species: {n_species}")
    print(f"Number of Compartments: {n_compartments}")
    print(reaction_data)

    # Split simulation_result per species for clarity
    species_grids = [simulation_result[:, i, :] for i in range(n_species)]
    species_names = [f"Species {i}" for i in range(n_species)]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

    # Total mass function
    def total_mass(grid):
        return np.sum(grid, axis=1)  # sum over compartments for each timepoint

    species_total_mass = [total_mass(grid) for grid in species_grids]

    # Animation setup
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = []
    for grid, color, name in zip(species_grids, colors, species_names):
        bar_group = ax.bar(space, grid[0, :], width=h, color=color,align = 'edge', alpha=0.7, label=name)
        bars.append(bar_group)

    ax.set_xlabel("Spatial Domain")
    ax.set_ylabel("Number of Molecules")
    ax.set_title("SSA Simulation")
    ax.set_xlim(0, domain_length)
    ax.set_ylim(0, max(grid.max() for grid in species_grids)*1.1)
    ax.legend()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def update(frame):
        for bar_group, grid in zip(bars, species_grids):
            for bar, height in zip(bar_group, grid[frame, :]):
                bar.set_height(height)
        time_text.set_text(f"Time: {time_vector[frame]:.2f}")
        return (*[b for group in bars for b in group], time_text)

    ani = FuncAnimation(fig, update, frames=range(0, len(time_vector), 1), interval=50)
    plt.show()

    # Plot total mass per species
    plt.figure(figsize=(8, 6))
    for mass, color, name in zip(species_total_mass, colors, species_names):
        plt.plot(time_vector, mass, color=color, label=f"{name} Total Mass")
    plt.xlabel("Time")
    plt.ylabel("Total Molecules")
    plt.title("Total Molecules per Species Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    main(ssa_path)
