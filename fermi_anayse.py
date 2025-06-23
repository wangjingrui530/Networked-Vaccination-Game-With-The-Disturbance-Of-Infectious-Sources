import matplotlib.pyplot as plt
import math
import numpy as np

def fermi(cost_i, cost_j, k):
    prob = 1 / (1 + math.exp((cost_i - cost_j) / k))
    return prob

vac_cost = np.arange(0,1.00001,0.001)

prob = []
for cost in vac_cost:
    prob.append(fermi(-cost, -1, k = 1))
plt.figure()
plt.title('vac learn non-vac prob k=1')
plt.plot(vac_cost, prob)
plt.show()

prob = []
for cost in vac_cost:
    prob.append(fermi(-cost, -1, k = 0.1))
plt.figure()
plt.title('vac learn non-vac prob k=0.1')
plt.plot(vac_cost, prob)
plt.show()

prob = []
for cost in vac_cost:
    prob.append(fermi(-1, -cost, k = 1))
plt.figure()
plt.title('non-vac learn vac prob k=1')
plt.plot(vac_cost, prob)
plt.show()

prob = []
for cost in vac_cost:
    prob.append(fermi(-1, -cost, k = 0.1))
plt.figure()
plt.title('non-vac learn vac prob k=0.1')
plt.plot(vac_cost, prob)
plt.show()
