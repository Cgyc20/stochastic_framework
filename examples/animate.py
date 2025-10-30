import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Path to saved SSA data


ssa_path = "data/SSA_data.npz"
# ---- USER OPTIONS ----
PLOT_CONCENTRATION = True   # True → show concentrations; False → show molecule counts
RESCALE_DOMAIN = True       # True → scale x-axis to [0, 1]
OMEGA =  10                # Must match your SSA simulation
# -----------------------

def main():
    sns.set_theme(style="whitegrid")

    # Load SSA data
    data = np.load(ssa_path, allow_pickle=True)
    simulation_result = data["simulation_result"]  # shape: (time, species, compartments)
    time_vector = data["timevector"]
    space = data["space"]
    domain_length = float(data["domain_length"])
    total_time = float(data["total_time"])
    timestep = float(data["timestep"])
    h = float(data["h"])
    n_species = int(data["n_species"])
    n_compartments = int(data["n_compartments"])
    reaction_data = data["reaction_data"].item()

    print("\n--- SSA DATA INFO ---")
    print(f"Simulation shape: {simulation_result.shape}")
    print(f"Domain length: {domain_length}")
    print(f"Compartment size h: {h}")
    print(f"Number of species: {n_species}, compartments: {n_compartments}")
    print(reaction_data)

    # ---- RESCALING ----
    if RESCALE_DOMAIN:
        print("Rescaling domain to [0, 1]...")
        space = space / domain_length
        h_scaled = h / domain_length
    else:
        h_scaled = h

    if PLOT_CONCENTRATION:
        print("Rescaling molecule counts to concentrations...")
        simulation_result = simulation_result / (OMEGA * h)
        ylabel = "Concentration"
    else:
        ylabel = "Number of Molecules"

    # Split per species
    species_grids = [simulation_result[:, i, :] for i in range(n_species)]
    species_names = [f"Species {i}" for i in range(n_species)]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

    # ---- TOTAL INTEGRAL / MASS ----
    def total_quantity(grid):
        if PLOT_CONCENTRATION:
            return np.sum(grid * h_scaled, axis=1)  # integrate conc × dx
        else:
            return np.sum(grid, axis=1)

    species_totals = [total_quantity(grid) for grid in species_grids]

    # ---- ANIMATION ----
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = []
    for grid, color, name in zip(species_grids, colors, species_names):
        bar_group = ax.bar(space, grid[0, :], width=h_scaled, align="edge",
                           color=color, alpha=0.7, label=name)
        bars.append(bar_group)

    ax.set_xlabel("Spatial Domain (scaled)" if RESCALE_DOMAIN else "Spatial Domain")
    ax.set_ylabel(ylabel)
    ax.set_title(f"SSA Simulation ({'Concentration' if PLOT_CONCENTRATION else 'Molecules'})")
    ax.set_xlim(0, 1 if RESCALE_DOMAIN else domain_length)
    ax.set_ylim(0, max(grid.max() for grid in species_grids)*1.1)
    ax.legend()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')

    def update(frame):
        for bar_group, grid in zip(bars, species_grids):
            for bar, height in zip(bar_group, grid[frame, :]):
                bar.set_height(height)
        time_text.set_text(f"Time: {time_vector[frame]:.2f}")
        return (*[b for group in bars for b in group], time_text)

    ani = FuncAnimation(fig, update, frames=range(0, len(time_vector), 1), interval=50)
    plt.show()



    #calculate analytic mass over time

    # r_birth = 0.5
    # r_death = 0.8
    # initial_concentration = 30 / (OMEGA* h)  # assuming initial molecule count of 30 per compartment

    # analytic_conc = initial_concentration * np.exp((r_birth-r_death)*time_vector)
    # ---- TOTALS PLOT ----
    plt.figure(figsize=(8, 6))
    for total, color, name in zip(species_totals, colors, species_names):
        plt.plot(time_vector, total, color=color, label=f"{name} Total")
    # plt.plot(time_vector, analytic_conc if PLOT_CONCENTRATION else analytic_conc * OMEGA * h * n_compartments,
    #          'k--', label="Analytic Solution", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Integrated " + ("Concentration" if PLOT_CONCENTRATION else "Molecule Count"))
    plt.title(f"Total {'Concentration' if PLOT_CONCENTRATION else 'Mass'} Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":

    main()
