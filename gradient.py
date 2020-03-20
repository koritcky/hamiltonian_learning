import numpy as np
from modules.smc import Particle


def conjugatet_angles(original_angles):
    n_spins = len(original_angles)

    # I think this can be realized more elegantly, but I don't understand how
    even_angles = np.copy(original_angles)
    odd_angles = np.copy(original_angles)
    for i in range(n_spins):
        if i % 2 == 0:
            even_angles[i] += np.pi / 2
        else:
            odd_angles[i] += np.pi / 2
    return even_angles, odd_angles


def measurements_for_gradient(rho: Particle, original_angles, even_angles, odd_angles):
    n_spins = rho.n_spins

    singles_orig, correlators_orig = rho.measure(original_angles)  # //////
    singles_even, correlators_even = rho.measure(even_angles)  # \/\/\/
    singles_odd, correlators_odd = rho.measure(odd_angles)  # /\/\/\

    singles_conjugated = np.zeros(n_spins)  # \ \ \ \ \ \
    correlators_conjugated_1 = np.zeros(n_spins - 1)  # \/ \/ \/ \/ \/
    correlators_conjugated_2 = np.zeros(n_spins - 1)  # /\ /\ /\ /\ /\
    for i in range(n_spins):
        if i % 2 == 0:
            singles_conjugated[i] = singles_even[i]
            if i < n_spins:
                correlators_conjugated_1[i] = correlators_even[i]
                correlators_conjugated_2[i] = correlators_odd[i]
        else:
            singles_conjugated[i] = singles_odd[i]
            if i < n_spins:
                correlators_conjugated_1[i] = correlators_odd[i]
                correlators_conjugated_2[i] = correlators_even[i]
    return singles_orig, singles_conjugated, correlators_orig, correlators_conjugated_1, correlators_conjugated_2


n_spins = 5
x = np.random.rand(n_spins) * 2 - 1
y = np.random.rand(n_spins) * 2 - 1
xx = np.random.rand(n_spins - 1) * 2 - 1
particle = Particle(weight=0.5, n_spins=5, beta=0.3, x=x, y=y, xx=xx)
particle.get_density_mat()

orig_angles = np.random.rand(n_spins, 2)
even_angles, odd_angles = conjugatet_angles(orig_angles)

s_o, c_o, s_c, c_c = measurements_for_gradient(particle, orig_angles, even_angles, odd_angles)
print(s_o, c_o, s_c, c_c)
