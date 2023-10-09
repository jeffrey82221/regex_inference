import pandas as pd

table = pd.read_parquet('raw.parquet')
table = table.sort_values('mean')
print(table)
table.to_csv('order_by_mean.csv')

table = table.sort_values('std')
print(table)
table.to_csv('order_by_std.csv')

table['mean-2std'] = table['mean'] - table['std'] * 2

table = table.sort_values('mean-2std')
table.to_csv('order_by_mean-2std.csv')

table['mean+2std'] = table['mean'] + table['std'] * 2

table = table.sort_values('mean+2std')
table.to_csv('order_by_mean+2std.csv')

from matplotlib import pylab as plt
table.plot(
   x='std', 
   y='mean', 
   kind='scatter'
)
plt.show()