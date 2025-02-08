import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

oj = pd.read_csv('/Users/garysutton/PycharmProjects/orangejuice.csv',
                 usecols = [1, 2, 3,])
print(oj.info())
print(oj.head())

# calculate proportion of defectives (p)
oj['proportion'] = oj['D'] / oj['size']

# calculate overall proportion defective (p-bar)
p_bar = oj['proportion'].mean()

# calculate control limits
oj['sigma'] = np.sqrt((p_bar * (1 - p_bar)) / oj['size'])
oj['UCL'] = p_bar + 3 * oj['sigma']
oj['LCL'] = p_bar - 3 * oj['sigma']

# plot the p-chart
plt.figure()
plt.plot(oj['sample'], oj['proportion'],
         marker = 'o',
         linestyle = '-',
         label = 'Proportion Defective')
plt.axhline(y = p_bar,
            color = 'r', linestyle = '--',
            label = 'p-bar (Mean)')
plt.plot(oj['sample'], oj['UCL'],
         color = 'g', linestyle = '--',
         label = 'UCL')
plt.plot(oj['sample'], oj['LCL'],
         color = 'g', linestyle = '--',
         label = 'LCL')
plt.fill_between(oj['sample'], oj['LCL'], oj['UCL'],
                 color = 'g', alpha = 0.1)
plt.title('p-Chart')
plt.xlabel('Sample')
plt.ylabel('Proportion Defective')
plt.legend(loc = 'upper right',
           bbox_to_anchor = (1, .95))
plt.grid(True)
plt.show()

# plot the np-chart
np_bar = oj['D'].mean()
UCL_np = np_bar + 2 * np.sqrt(np_bar)
LCL_np = np_bar - 2 * np.sqrt(np_bar)

plt.figure()
plt.plot(oj['sample'], oj['D'],
         marker = 'o',
         linestyle = '-',
         label = 'Number of Defective Items')
out_of_control = (oj['D'] > UCL_np) | (oj['D'] < LCL_np)
plt.plot(oj['sample'][out_of_control],
         oj['D'][out_of_control],
         marker = 'o', color = 'r', linestyle = 'None',
         label='Out of Control')
plt.axhline(y = np_bar,
            color = 'r', linestyle = '--',
            label = 'np-bar (Mean)')
plt.axhline(y = UCL_np,
            color = 'g', linestyle = '--',
            label = 'UCL')
plt.axhline(y = LCL_np,
            color = 'g', linestyle = '--',
            label = 'LCL')
plt.fill_between(oj['sample'],
                 LCL_np, UCL_np,
                 color = 'g', alpha = 0.1)
plt.title('np-Chart')
plt.xlabel('Sample')
plt.ylabel('Number of Defective Items')
plt.legend(loc = 'upper left',
           bbox_to_anchor = (0.1, 0.9))
plt.grid(True)
plt.show()

# plot the c-chart
np.random.seed(0)
units = 20
defects = np.random.poisson(lam = 4, size = units)

c_bar = np.mean(defects)
UCL = c_bar + 3 * np.sqrt(c_bar)
LCL = max(0, c_bar - 3 * np.sqrt(c_bar))

plt.figure()
plt.plot(range(1, units + 1), defects,
         marker = 'o', linestyle = '-', color = 'b',
         label = 'Number of Defects')
plt.axhline(c_bar,
            color = 'r', linestyle = '--',
            label = 'Center Line')
plt.axhline(UCL,
            color = 'g', linestyle = '--',
            label = 'UCL')
plt.axhline(LCL,
            color='g', linestyle = '--',
            label = 'LCL')
plt.title('c-Chart')
plt.xlabel('Unit Number')
plt.ylabel('Number of Defects')
plt.xticks(range(1, units + 1))
plt.legend(loc = 'upper right',
           bbox_to_anchor = (1, .95))
plt.grid(True)
plt.show()

# plot the g-chart
data = {
    'defect_sequence': np.arange(1, 21),
    'units_between_defects': [50, 30, 45, 60, 55, 35, 40,
                              70, 65, 80, 75, 50, 85, 90,
                              95, 60, 55, 100, 105, 110]
}
df = pd.DataFrame(data)

g_bar = df['units_between_defects'].mean()
UCL = g_bar + 3 * np.sqrt(g_bar * (g_bar + 1))
LCL = max(g_bar - 3 * np.sqrt(g_bar * (g_bar + 1)), 0)

plt.figure()
plt.plot(df['defect_sequence'], df['units_between_defects'],
         marker = 'o', linestyle = '-',
         label = 'Units Between Defects')
plt.axhline(y = g_bar,
            color = 'r', linestyle = '--',
            label = 'g-bar (Mean)')
plt.axhline(y = UCL,
            color = 'g', linestyle = '--',
            label = 'UCL')
plt.axhline(y = LCL,
            color = 'g', linestyle = '--',
            label = 'LCL')
plt.fill_between(df['defect_sequence'], LCL, UCL,
                 color = 'g', alpha = 0.1)
plt.title('g-Chart')
plt.xlabel('Defect Sequence')
plt.ylabel('Units Between Defects')
plt.legend()
plt.grid(True)
plt.show()

pistons = pd.read_csv('/Users/garysutton/PycharmProjects/pistonrings.csv',
                 usecols = [1, 2])
print(pistons.info())
pistons['sample'].value_counts()
print(pistons.head(10))
print(pistons.tail())
stats = pistons['diameter'].describe()
print(stats)

# plot the xbar-chart
x_bar = pistons.groupby('sample')['diameter'].mean()
overall_mean = x_bar.mean()
sigma_x_bar = x_bar.std() / np.sqrt(len(x_bar))
UCL = overall_mean + 3 * sigma_x_bar
LCL = overall_mean - 3 * sigma_x_bar

plt.figure()
plt.plot(x_bar.index, x_bar.values,
         marker = 'o', linestyle = '-', color = 'b',
         label = 'Sample Means')
plt.axhline(overall_mean, color = 'r', linestyle = '--',
            label = 'Center Line')
plt.axhline(UCL, color='g', linestyle='--',
            label = 'UCL')
plt.axhline(LCL, color = 'g', linestyle = '--',
            label = 'LCL')
plt.fill_between(pistons['sample'], LCL, UCL,
                 color = 'g', alpha = 0.1)
plt.title('x-bar Chart')
plt.xlabel('Sample')
plt.ylabel('Mean Diameter')
plt.legend(loc = 'upper left')
plt.grid(True)
plt.show()

# plot the r-chart
r_values = pistons.groupby('sample')['diameter'].apply(lambda x: x.max() - x.min())
r_bar = r_values.mean()
sigma_r = r_values.std()
UCL = r_bar + 3 * sigma_r
LCL = max(r_bar - 3 * sigma_r, 0)

plt.figure()
plt.plot(r_values.index, r_values.values,
         marker = 'o', linestyle = '-', color = 'b',
         label = 'Sample Ranges')
plt.axhline(r_bar, color = 'r', linestyle = '--',
            label = 'Center Line')
plt.axhline(UCL, color = 'g', linestyle = '--',
            label = 'UCL')
plt.axhline(LCL, color = 'g', linestyle = '--',
            label = 'LCL')
plt.fill_between(r_values.index, LCL, UCL,
                 color = 'g', alpha = 0.1)
plt.title('r-Chart')
plt.xlabel('Sample')
plt.ylabel('Range of Diameters')
plt.legend(loc = 'lower right',
           bbox_to_anchor = (0.9, 0.1))
plt.grid(True)
plt.show()

# plot the s-chart
s = pistons.groupby('sample')['diameter'].std()
s_bar = s.mean()
c4 = np.sqrt(2 / (len(s) - 1))
UCL = s_bar + 3 * (s_bar / c4)
LCL = s_bar - 3 * (s_bar / c4)

plt.figure()
plt.plot(s.index, s.values,
         marker = 'o', linestyle = '-', color = 'b',
         label = 'Sample Standard Deviations')
plt.axhline(s_bar, color = 'r', linestyle = '--',
            label = 'Center Line')
plt.axhline(UCL, color = 'g', linestyle = '--',
            label = 'UCL')
plt.axhline(LCL, color = 'g', linestyle = '--',
            label = 'LCL')
plt.fill_between(s.index, LCL, UCL,
                 color = 'g', alpha = 0.1)
plt.title('s-Chart')
plt.xlabel('Sample')
plt.ylabel('Standard Deviation')
plt.legend(loc = 'upper right',
           bbox_to_anchor = (1, 0.95))
plt.grid(True)
plt.show()

# plot the I-MR Chart
individuals = pistons['diameter']
moving_range = individuals.diff().abs()

I_bar = individuals.mean()
MR_bar = moving_range[1:].mean()

UCL_I = I_bar + 3 * individuals.std()
LCL_I = I_bar - 3 * individuals.std()

D4 = 3.267
D3 = 0
UCL_MR = D4 * MR_bar
LCL_MR = D3 * MR_bar

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(pistons['sample'], individuals,
         marker = 'o', linestyle = '-', color = 'b',
         label = 'Individual Measurements')
plt.axhline(I_bar, color = 'r', linestyle = '--',
            label = 'Center Line')
plt.axhline(UCL_I, color = 'g', linestyle = '--',
            label = 'UCL')
plt.axhline(LCL_I, color = 'g', linestyle = '--',
            label = 'LCL')
plt.fill_between(pistons['sample'], LCL_I, UCL_I,
                 color = 'g', alpha = 0.1)
plt.title('I-MR Chart')
plt.xlabel('Sample')
plt.ylabel('Diameter')
plt.legend(loc = 'lower right')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(pistons['sample'][1:], moving_range[1:],
         marker = 'o', linestyle = '-', color = 'b',
         label = 'Moving Range')
plt.axhline(MR_bar, color = 'r', linestyle = '--',
            label = 'Center Line')
plt.axhline(UCL_MR, color = 'g', linestyle = '--',
            label = 'UCL')
plt.axhline(LCL_MR, color = 'g', linestyle = '--',
            label = 'LCL')
plt.fill_between(pistons['sample'][1:], LCL_MR, UCL_MR,
                 color = 'g', alpha = 0.1)
plt.xlabel('Sample')
plt.ylabel('Moving Range')
plt.legend(loc = 'upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# plot the EWMA chart
lambda_ = 0.3
n = len(pistons)

ewma = np.zeros(n)
ewma[0] = pistons['diameter'][0]

for i in range(1, n):
    ewma[i] = lambda_ * pistons['diameter'][i] + (1 - lambda_) * ewma[i - 1]

ewma_center = np.mean(ewma)
sigma = np.std(pistons['diameter'])
UCL_ewma = ewma_center + 3 * sigma * np.sqrt(lambda_ / (2 - lambda_))
LCL_ewma = ewma_center - 3 * sigma * np.sqrt(lambda_ / (2 - lambda_))

plt.figure()
plt.plot(pistons['sample'], ewma,
         marker = 'o', linestyle = '-', color = 'b',
         label = 'EWMA')
plt.axhline(ewma_center, color = 'r', linestyle = '--',
            label = 'Center Line')
plt.axhline(UCL_ewma, color='g', linestyle='--',
            label = 'UCL')
plt.axhline(LCL_ewma, color='g', linestyle='--',
            label = 'LCL')
plt.fill_between(pistons['sample'], LCL_ewma, UCL_ewma,
                 color = 'g', alpha = 0.1)
plt.title('EWMA Chart')
plt.xlabel('Sample')
plt.ylabel('EWMA of Diameter')
plt.legend(loc = 'upper right')
plt.grid(True)
plt.show()





































