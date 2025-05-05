import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Beispiel-Parameter (alpha = Erfolge + 1, beta = Fehlschläge + 1)
parameter_sets = [
    (0, 0),    # Gleichverteilung
    (5, 1),    # viele Erfolge
    (1, 5),    # viele Misserfolge
    (2, 2),    # symmetrisch, unsicher
    (10, 10),  # symmetrisch, sicher
    (30, 5),   # starker Erfolg
]

x = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))

for alpha, beta_param in parameter_sets:
    y = beta.pdf(x, alpha, beta_param)
    plt.plot(x, y, label=f'α={alpha}, β={beta_param}')

plt.title('Beta-Verteilungen für verschiedene Parameter')
plt.xlabel('Wahrscheinlichkeit')
plt.ylabel('Dichte')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
