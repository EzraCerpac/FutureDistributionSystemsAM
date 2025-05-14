import numpy as np
import matplotlib.pyplot as plt

# Given parameters
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
mur = 50000            # Relative permeability of the core
mu = mu0 * mur         # Absolute permeability (H/m)
sigma = 1e7            # Conductivity (S/m)

# Frequencies in Hz
frequencies = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

# Angular frequency
omega = 2 * np.pi * frequencies

# Skin depth calculation
delta = np.sqrt(2 / (mu * omega * sigma))

# Plot
plt.figure()
plt.plot(frequencies, delta)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Skin depth δ (m)')
plt.title('Skin Depth vs Frequency')
plt.grid(True)
plt.show()
