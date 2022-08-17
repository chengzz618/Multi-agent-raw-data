import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

T = 100

d1 = pd.read_csv('0.2_bit_1.csv',index_col = 'episode')
d2 = pd.read_csv('0.2_bit_2.csv',index_col = 'episode')
d3 = pd.read_csv('0.2_bit_3.csv',index_col = 'episode')
d4 = pd.read_csv('0.2_bit_4.csv',index_col = 'episode')

d1['1'] = d1['reward'].ewm(span = T).mean()
d2['2'] = d2['reward'].ewm(span = T).mean()
d3['3'] = d3['reward'].ewm(span = T).mean()
d4['4'] = d4['reward'].ewm(span = T).mean()

d1['1'].plot(label='bit = 1')
d2['2'].plot(label='bit = 2')
d3['3'].plot(label='bit = 3')
d4['4'].plot(label='bit = 4')

plt.title('Training Results with Different bit of Communication ( Sigma=0.2; Episode_Max=5000 )')
plt.ylabel('Reward')
plt.legend()

plt.show()

