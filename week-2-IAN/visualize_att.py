import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

str1 = 'components'
l = [1.0]
str = 'I was also informed that the _ of the Mac Book were dirty .'
b = [0.0156300850212574, 0.01564560830593109, 0.015646034851670265, 0.01565961167216301, 0.017852718010544777,
 0.11114897578954697, 0.0, 0.11548236757516861, 0.11548243463039398, 0.11549139022827148, 0.11549157649278641,
 0.11549157649278641, 0.11549157649278641, 0.11548605561256409]
"""
by copying the contexts in laptops_text_processed.txt/restautants_text_processed.txt to 
str1 (target), l (attention weights for target), str (context), b(attention weights for context)
this .py program creates heatmap to visualize attention weights of target-context
"""
sns.set()
l = np.array(l)
b = np.array(b)
l = np.expand_dims(l, axis=0)
b = np.expand_dims(b, axis=1)
l = np.multiply(b, l)
d = pd.DataFrame(l, index=str.split(), columns=str1.split())
fig = plt.figure()
sns_plot = sns.heatmap(d, cmap='YlGnBu',linewidth=0.003, square=True)
plt.show()
