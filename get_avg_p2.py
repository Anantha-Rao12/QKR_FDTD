import numpy as np
import matplotlib.pyplot as plt
from qpkr import QPKR


basis1 = 512
total_time = 100
params1 = [3, 4 * np.pi, 0, 0, 0, 0, 0]
params2 = [3, 1, 0, 0, 0, 0, 0]

qpkr1 = QPKR(basis1, total_time, params1)
qpkr2 = QPKR(basis1, total_time, params2)

p2_1, psi_t = qpkr1.evolve("random", 0.01)
p2_2, psi_t = qpkr2.evolve(form="QPKR")

plt.plot(p2_1)
plt.plot(p2_2)
plt.show()
