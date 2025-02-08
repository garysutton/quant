import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

### code not in book; only outputs ###
df1 = pd.read_csv('/Users/garysutton/PycharmProjects/sample_data.csv',
                   usecols = [0, 1])
print(df1)

y = df1['y']
x1 = df1[['x']]
x = sm.add_constant(x1)

fit_test1 = sm.OLS(y, x).fit()
print(fit_test1.summary())

fig, ax = plt.subplots()
ax.scatter(df1['x'], df1['y'])
ax.set_xlim(150, 250)
ax.set_ylim(150, 350)
for i, (x_val, y_val) in enumerate(zip(df1['x'], df1['y'])):
    plt.text(x_val, y_val, f'({x_val}, {y_val})',
             fontsize = 8, ha = 'center', va = 'bottom')
y = -12.1820 + (1.3413 * x)
ax.plot(x, y, linewidth = 2, color = 'red')
ax.set_title('y = -12.1820 + (1.3413 * x)\nR-squared = 0.88',
             fontweight = 'bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
highlight_index = 4
highlight_color = 'orange'
highlight_size = 100
ax.scatter(df1['x'][highlight_index], df1['y'][highlight_index],
           color = highlight_color, s = highlight_size)
plt.grid()
plt.show()

df2 = pd.read_csv('/Users/garysutton/PycharmProjects/sample_data.csv',
                   usecols = [2, 3])
print(df2)

y = df2['y1']
x1 = df2[['x1']]
x = sm.add_constant(x1)

fit_test2 = sm.OLS(y, x).fit()
print(fit_test2.summary())

fig, ax = plt.subplots()
ax.scatter(df2['x1'], df2['y1'])
ax.set_xlim(140, 250)
ax.set_ylim(80, 140)
y = 190.6050 - (0.3924 * x)
ax.plot(x, y, linewidth = 2, color = 'red')
ax.set_title('A negative relationship between x and y',
             fontweight = 'bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.grid()
plt.show()

df3 = pd.read_csv('/Users/garysutton/PycharmProjects/sample_data.csv',
                   usecols = [4, 5])
print(df3)

y = df3['y2']
x1 = df3[['x2']]
x = sm.add_constant(x1)

fit_test3 = sm.OLS(y, x).fit()
print(fit_test3.summary())

fig, ax = plt.subplots()
ax.scatter(df3['x2'], df3['y2'])
ax.set_xlim(140, 250)
ax.set_ylim(80, 280)
y = 179.5043 - (0.0682 * x)
ax.plot(x, y, linewidth = 2, color = 'red')
ax.set_title('A neutral relationship between x and y',
             fontweight = 'bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.grid()
plt.show()

df4 = df1.loc[df1.index != 4].copy()
print(df4)

y = df4['y']
x1 = df4[['x']]
x = sm.add_constant(x1)

fit_test4 = sm.OLS(y, x).fit()
print(fit_test4.summary())

fig, ax = plt.subplots()
ax.scatter(df4['x'], df4['y'])
ax.set_xlim(150, 250)
ax.set_ylim(150, 350)
for i, (x_val, y_val) in enumerate(zip(df4['x'], df4['y'])):
    plt.text(x_val, y_val, f'({x_val}, {y_val})',
             fontsize = 8, ha = 'center', va = 'bottom')
y = -8.9857 + (1.3032 * x)
ax.plot(x, y, linewidth = 2, color = 'red')
ax.set_title('y = -8.9857 + (1.3032 * x)\nR-squared = 0.95',
             fontweight = 'bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.grid()
plt.show()

###############

mds = pd.read_csv('/Users/garysutton/PycharmProjects/mds.csv')
print(mds.info())
print(mds)
print(mds.describe())

print(stats.shapiro(mds.stage1))
print(stats.shapiro(mds.stage2))

upper_threshold = 194.55 + (3 * 29.62)
lower_threshold = 194.55 - (3 * 29.62)
exceed_threshold = mds[(mds['stage1'] > upper_threshold) | \
                       (mds['stage1'] < lower_threshold)]
print(exceed_threshold)

upper_threshold = 251.45 + (3 * 51.11)
lower_threshold = 251.45 - (3 * 51.11)
exceed_threshold = mds[(mds['stage2'] > upper_threshold) | \
                       (mds['stage2'] < lower_threshold)]
print(exceed_threshold)

y = mds['stage2']
x = mds['stage1']
x = sm.add_constant(x)

lm = sm.OLS(y, x).fit()
print(lm.summary())

plt.scatter(mds['stage1'], mds['stage2'])
plt.xlim(150, 260)
plt.ylim(150, 350)
yhat = -3.078 + (1.3083 * x)
plt.plot(x, yhat, linewidth = 2, color = 'red')
plt.title('2021 Marathon Des Sables - Top 20 Male Finishers\n'
          'y = -3.078 + (1.3083 * x)\n'
          'R-squared = 0.575', fontweight = 'bold')
plt.xlabel('Stage 1 Running Time (min)')
plt.ylabel('Stage 2 Running Time (min)')
plt.grid()
plt.show()

SST = np.sum((mds['stage2'] - np.mean(mds['stage2'])) ** 2)
print(SST)

SSR = np.sum((lm.predict(x) - np.mean(mds['stage2'])) ** 2)
print(SSR)

print(lm.fittedvalues)

R2 = SSR / SST
print(R2)

SSE = SST - SSR
print(SSE)

SST_new = SSR + SSE
print(SST_new)

SSE = np.sum(lm.resid ** 2)
print(SSE)

R2 = (1 - (SSE / SST))
print(R2)

F = (SSR / 1) / (SSE / (20 - 1 - 1))
print(F)

stage1_sum = mds['stage1'].sum()
print(stage1_sum)

stage2_sum = mds['stage2'].sum()
print(stage2_sum)

plt.scatter(lm.fittedvalues, lm.resid)
plt.axhline(y = 0, color = 'red', linestyle = '--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

print(durbin_watson(lm.resid))

lm_BreuschPagan = het_breuschpagan(lm.resid, lm.model.exog)
print(lm_BreuschPagan[1])

stats.probplot(lm.resid, dist = 'norm', plot = plt)
plt.gca().get_lines()[1].set_linestyle('--')
plt.title('Q-Q Plot')
plt.show()

print(stats.jarque_bera(lm.resid))




