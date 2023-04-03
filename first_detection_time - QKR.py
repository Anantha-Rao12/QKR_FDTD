import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt


def get_first_ht(k_mu, k_sigma, tau, timesteps):
    """Given the hitting strength and measurement interval,
    compute the amplitude of first detection and first hitting time
    distribution"""
    kappa = np.random.randn(timesteps) * k_sigma + k_mu
# np.random.standard_cauchy
    get_unitary = lambda k, n, tau: jv(0, 2 * k * n * tau)
    unitary = np.array(
        [get_unitary(k_i, n_i, tau) for (k_i, n_i) in zip(kappa, range(timesteps))]
    )
    psi = np.ones(timesteps)

    for i in range(timesteps):
        psi[i] = unitary[i] - sum([unitary[i - j] * psi[j] for j in range(i - 1)])

    return psi, np.abs(psi) ** 2


tau = np.pi / 2
timesteps = 1000

k_mu = 1
k_sigma = 1e-03


psi1, F1 = get_first_ht(k_mu=k_mu, k_sigma=0, tau=tau, timesteps=timesteps)
psi2, F2 = get_first_ht(k_mu=k_mu, k_sigma=k_sigma, tau=tau, timesteps=timesteps)

fig, ax = plt.subplots()

ax.plot(
    np.log10(range(1, timesteps + 1)),
    np.log10(F1),
    "bo",
    markersize=5,
    label=r"$\kappa$:3", alpha=0.5
)
ax.plot(
    np.log10(range(1, timesteps + 1)),
    np.log10(F2),
    "ro",
    markersize=5,
    label=r"$\kappa$:N($\mu$=3, $\sigma$=0.01)", alpha=0.5
)
ax.legend(fontsize=15)

ax.set_xlabel("Measurement time", fontsize=22)
ax.set_ylabel("First hitting probability ($F_n$)", fontsize=22)
ax.set_title(r"First hitting time distribution $\tau = \pi/2$", fontsize=22)
ax.set(xticks=np.arange(0, 4), yticks=np.arange(-14, 0, 4))
ax.set_xticklabels(["1", "10", "100", "1000"])
ax.set_yticklabels(["1e-14", "1e-10", "1e-6", "1e-2"])

plt.show()
