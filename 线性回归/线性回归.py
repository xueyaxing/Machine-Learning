import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([0.5,0.5,0.8,1.1,1.4])
y = np.array([5.0,5.5,6.0,6.8,7.0])
plt.grid(linestyle=':')
plt.scatter(x,y,s=60,color='dodgerblue',label='Samples')
plt.show()
