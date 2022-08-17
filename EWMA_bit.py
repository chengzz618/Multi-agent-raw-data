import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

T = 100

d1 = pd.read_csv('bit1_0.4.csv',index_col = 'episode')
d2 = pd.read_csv('bit2_0.4.csv',index_col = 'episode')
d3 = pd.read_csv('bit3_0.4.csv',index_col = 'episode')
d4 = pd.read_csv('bit4_0.4.csv',index_col = 'episode')
d5 = pd.read_csv('bit5_0.4.csv',index_col = 'episode')
d6 = pd.read_csv('bit10_0.4.csv',index_col = 'episode')
d7 = pd.read_csv('bit50_0.4.csv',index_col = 'episode')

d1['1'] = d1['reward'].ewm(span = T).mean()
d2['2'] = d2['reward'].ewm(span = T).mean()
d3['3'] = d3['reward'].ewm(span = T).mean()
d4['4'] = d4['reward'].ewm(span = T).mean()
d5['5'] = d5['reward'].ewm(span = T).mean()
d6['6'] = d6['reward'].ewm(span = T).mean()
d6['7'] = d7['reward'].ewm(span = T).mean()

d1['1'].plot(label='bit = 1')
d2['2'].plot(label='bit = 2')
d3['3'].plot(label='bit = 3')
d4['4'].plot(label='bit = 4')
d5['5'].plot(label='bit = 5')
d6['6'].plot(label='bit = 10')
d6['7'].plot(label='bit = 50')

plt.title('Training Results with Different bit of Communication ( Sigma=0.4; Span = 100; Episode_Max=5000 )')
plt.ylabel('Reward')
plt.legend()

plt.show()

