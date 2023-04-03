import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt


def get_unitary(kappa: float, hbar: float, tau: float, n: int, basis: int) -> float:
    """Returns the (0,0) or the (n/2, n/2) element of the unitary operator of a QKR
    kappa: distribution of kicking strength
    hbar: effective planck's contant
    tau: measurement time (multiple after kicking)
    n: number of kicks"""
    return np.exp(-1j * hbar * n * basis ** 2 / 8) * jv(0, 2 * kappa * n * tau)


def get_QKR_first_hittingtime(k_mu, k_sigma, tau, hbar, basis, timesteps):
    """Given the hitting strength and measurement interval
    compute the amplitude of first detection and first hitting time
    distribution"""
    kappa = np.random.randn(timesteps) * k_sigma + k_mu
    unitaries = np.array(
        [
            get_unitary(k_i, hbar, tau, n_i, basis)
            for (k_i, n_i) in zip(kappa, range(timesteps))
        ]
    )
    psi = np.ones(timesteps)

    for i in range(timesteps):
        psi[i] = unitaries[i] - sum([unitaries[i - j] * psi[j] for j in range(i - 1)])

    return psi, np.abs(psi) ** 2


tau = np.pi / 2
timesteps = 1000

k_mu = 3
k_sigma = 1e-02
hbar1 = 4 * np.pi
hbar2 = 1
basis = 1024


psi1, F1 = get_QKR_first_hittingtime(k_mu, 0, tau, hbar1, basis, timesteps)
psi2, F2 = get_QKR_first_hittingtime(k_mu, 0, tau, hbar1, basis, timesteps)

fig, ax = plt.subplots()

ax.plot(
    np.log10(range(1, timesteps + 1)),
    np.log10(F1),
    "bo",
    markersize=5,
    label=r"$\kappa$:%d: $\hbar_s$:%.2f" % (k_mu, hbar1),
)
ax.plot(
    np.log10(range(1, timesteps + 1)),
    np.log10(F2),
    "ro",
    markersize=5,
    label=r"$\kappa$:$\mu$=%d, $\hbar_s$:%.2f" % (k_mu, hbar2),
)
ax.legend(fontsize=15)

ax.set_xlabel("Measurement time", fontsize=22)
ax.set_ylabel("First hitting probability ($F_n$)", fontsize=22)
ax.set_title(r"First hitting time distribution $\tau = \pi/2$", fontsize=22)
ax.set(xticks=np.arange(0, 4), yticks=np.arange(-14, 0, 4))
ax.set_xticklabels(["1", "10", "100", "1000"])
ax.set_yticklabels(["1e-14", "1e-10", "1e-6", "1e-2"])

plt.show()
