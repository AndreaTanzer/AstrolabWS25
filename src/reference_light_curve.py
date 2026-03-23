import numpy as np
from gatspy.datasets import fetch_rrlyrae
from astroML.datasets import fetch_rrlyrae_templates

import plot

rrlyrae = fetch_rrlyrae_templates()
g = rrlyrae["103g"][:, 1]
r = rrlyrae["103r"][:, 1]
t = rrlyrae["103g"][:, 0]
V = np.mean(np.array([g, r]), axis=0)

plot.plot(data=[t, V, g, r], legend=["V", "g", "r"], showPlot=True, invert_y_axis=True)