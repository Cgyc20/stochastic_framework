Ah, got it! You want the README to just show **the reaction rates as they are used in the SSA**, without worrying about dividing by compartment size or Ω. Here’s a revised version:

---

# Stochastic Framework

**Stochastic Framework** is a Python package for simulating **stochastic reaction-diffusion systems** using the **Gillespie / Stochastic Simulation Algorithm (SSA)**. It supports multiple species, spatial compartments, diffusion, and arbitrary reactions.

This project started as a personal tool for exploring stochastic dynamics in chemical or biological systems. It’s designed for flexibility and clarity: you define your reactions, set up compartments, and simulate trajectories.

---

## Features

* Define reactions with arbitrary stoichiometry and rates.
* Simulate multiple species across a 1D spatial domain divided into compartments.
* Include **diffusion** between compartments for each species.
* Run multiple simulation repeats and compute average trajectories.
* Save results in `.npz` files for later analysis, plotting, or animation.
* Optional rescaling for plotting concentrations or normalizing the domain.

---

## Example Reactions

Here’s an example system used in tests:

| Reaction    | Rate |
| ----------- | ---- |
| 2U → 3U     | 1.0  |
| 2U → 2U + V | 2.0  |
| U + V → V   | 1.0  |
| V → ∅       | 1.0  |

> The rates listed are the **actual rates you pass to the simulation**. No need to manually divide by compartment size—SSA handles the stochastic execution.

---

## Quick Start

```python
from stochastic_framework import Reaction, SSA
import numpy as np

# 1. Define reactions
R = Reaction()
R.add_reaction({"U": 2}, {"U": 3}, 1.0)
R.add_reaction({"U": 2}, {"U": 2, "V": 1}, 2.0)
R.add_reaction({"U":1, "V":1}, {"V":1}, 1.0)
R.add_reaction({"V":1}, {}, 1.0)

# 2. Initialize SSA
ssa = SSA(R)

# 3. Set initial conditions
n_compartments = 10
initial_conditions = np.zeros((2, n_compartments), dtype=int)
initial_conditions[0,:] = 50  # U
initial_conditions[1,:] = 25  # V

ssa.set_conditions(
    n_compartments=n_compartments,
    domain_length=10.0,
    total_time=40.0,
    initial_conditions=initial_conditions,
    timestep=0.1,
    Macroscopic_diffusion_rates=[0.05, 1.25]
)

# 4. Run simulation and save results
average_output = ssa.run_simulation(n_repeats=5)
ssa.save_simulation_data("data/SSA_data.npz", simulation_result=average_output)
```

---

## Notes on Usage

* **1D only**: Currently supports a single spatial dimension divided into compartments.
* **Diffusion**: Each species can have its own diffusion rate between compartments.
* **Analysis and plotting**: The saved `.npz` file contains the simulation tensor, time vector, space vector, and reaction metadata. You can animate or integrate total mass over time.

---

This version keeps it simple and matches exactly how your SSA expects rates.

If you want, I can also **add a tiny “how to plot concentrations or normalize the domain” section”** in one paragraph, so the README is fully self-contained. Do you want me to do that?
