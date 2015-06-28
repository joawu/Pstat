import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Concentration -> Graph Mapping
buffers = ['2', '6']
mapping = [[500,          '8'],
           [250,         '10'],
           [125,         '11'],
           [62.5,        '12'],
           [31.25,       '13'],
           [15.625,      '15'],
           [15.625,      '14'],
           [7.8125,      '16'],
           [3.90625,     '17'],
           [1.953125,    '18'],
           [0.9765625,   '19'],
           [0.48828125,  '20'],
           [0.244140625, '21'],
           ]
mapping = pd.DataFrame(mapping)
mapping.columns = ['Concentration', 'GraphID']

# Empty list to hold each files data
df = []

# Find each filename in the folder
data_directory = 'data/'
for filename in os.listdir(data_directory):
    # Load the data into a pandas dataframe
    graph = pd.DataFrame.from_csv(data_directory + filename).reset_index()

    # Find the ID from the filename
    graph_id = filename.split(' ')[1].split('.')[0]

    # Make a pandas series and put it in the df list
    values = graph[' y Value']
    df.append(pd.Series(values.values, name=graph_id))

# Concat the list into a pandas dataframe
df = pd.concat(df, axis=1)

# Make a time column out of the index
df['Time'] = np.array(df.index) / 2000

# Only take data from the last 4 seconds
data = df.query('Time > 20')

# Filter out bad runs based on the standard deviation
maximum_voltage_std = .1
good_graphs = (data.std() < maximum_voltage_std).reset_index()
good_graphs.columns = ['GraphID', 'Good']

# Grab the IDs of the good runs
good_graph_ids = good_graphs.query('Good == True').GraphID.tolist()

# Calculate the buffer value from the two buffer runs
buffer_value = data[buffers].mean().mean()

# Find the mean currents for each graph and subtract out the buffer value
means = data[good_graph_ids].mean().reset_index()
means.columns = ['GraphID', 'Current']
means_adjusted = means.copy()
means_adjusted.Current -= buffer_value
means_adjusted.columns = ['GraphID', 'AdjustedCurrent']
sems = data[good_graph_ids].sem().reset_index()
sems.columns = ['GraphID', 'Error']

# Merge dataframes together
results = mapping.merge(means).merge(means_adjusted).merge(sems)
# results = results.good_graphs('Concentration').mean()
linear_regime = results.query('Concentration > 100')

# Fit the data in the linear regime
x = linear_regime.Concentration
y = linear_regime.AdjustedCurrent
X = sm.add_constant(x)
model = sm.OLS(y, X)
fit = model.fit()
# could also return stderr in each via fit.bse
m, b = fit.params[1], fit.params[0]

# Make the fit line
points = np.linspace(x.min(), x.max(), 2)

# Draw the plot
f, ax = plt.subplots()
linear_regime.plot(kind='scatter', x='Concentration', y='AdjustedCurrent', yerr='Error', ax=ax)
ax.plot(points, m * points + b)
plt.title('rsquared: ' + str(fit.rsquared))
plt.show()

# Python 3.4.3
# numpy 1.9.2
# pandas 0.16.2
# statsmodels 0.6.1
# matplotlib 1.4.3
