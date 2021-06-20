import matplotlib.pyplot as plt
import numpy as np

y = np.array([50, 62, 104])
mylabels = ["True Positives", "False Positives", "False Negatives"]
colors = ["yellowgreen","lightcoral","firebrick"]


def absolute_value(val):
    a  = y[ np.abs(y - val/100.*y.sum()).argmin() ]
    return a


plt.pie(y, colors = colors, autopct=absolute_value, shadow = True)
plt.legend(labels = mylabels)
plt.title('C. jejuni', fontsize=12)
plt.show()
