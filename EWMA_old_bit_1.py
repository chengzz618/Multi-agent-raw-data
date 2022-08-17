import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

T = 50

d1 = pd.read_csv('0.1_bit_1.csv',index_col = 'episode')
d2 = pd.read_csv('0.2_bit_1.csv',index_col = 'episode')
d3 = pd.read_csv('0.3_bit_1.csv',index_col = 'episode')
d4 = pd.read_csv('0.5_bit_1.csv',index_col = 'episode')
d5 = pd.read_csv('2_bit_1.csv',index_col = 'episode')
d6 = pd.read_csv('10_bit_1.csv',index_col = 'episode')
d7 = pd.read_csv('1000_bit_1.csv',index_col = 'episode')

d1['0.1'] = d1['reward'].ewm(span = T).mean()
d2['0.2'] = d2['reward'].ewm(span = T).mean()
d3['0.3'] = d3['reward'].ewm(span = T).mean()
d4['0.5'] = d4['reward'].ewm(span = T).mean()
d5['2'] = d5['reward'].ewm(span = T).mean()
d6['10'] = d6['reward'].ewm(span = T).mean()
d7['1000'] = d7['reward'].ewm(span = T).mean()

d1['0.1'].plot(label='sigma=0.1')
d2['0.2'].plot(label='sigma=0.2')
d3['0.3'].plot(label='sigma=0.3')
d4['0.5'].plot(label='sigma=0.5')
d5['2'].plot(label='sigma=2')
d6['10'].plot(label='sigma=10')
d7['1000'].plot(label='sigma=1000')

plt.title('Training Results with Different Noise Sigma of Communication ( Bit = 1; Episode_Max=5000 )')
plt.ylabel('Reward')
plt.legend()

plt.show()

