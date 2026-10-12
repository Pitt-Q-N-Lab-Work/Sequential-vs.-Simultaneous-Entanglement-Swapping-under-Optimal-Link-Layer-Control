"""Physical constants and simulation defaults shared by every module in src/."""
F_MIN = 0.8114         # six-state key-zero fidelity: below this 1 - H(F) < 0 and no key survives
TC_INT = 0.1           # link memory coherence time (s), default
TC_EXT = 2.0           # chain memory coherence time (s), default
L_ATT = 22.0           # fiber attenuation length (km)
C_FIBER = 200_000      # speed of light in fiber (km/s)
K = 0.9                # p_gen = K * exp(-L / L_ATT)
N_ROUNDS = 10_000_000  # cap on end-to-end pairs per trial
N_TRIALS = 10
BUFFER_SIZE = 20
KAPPA = 1.0            # chain memory upgrade factor, pinned to 1 in this paper
