import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import benford as bf
import scipy.stats as stats
from scipy.stats import kurtosis
from scipy.stats import skew

df1 = pd.DataFrame({'d': list(range(1, 10))})
print(df1)

df1 = df1.assign(benford = round(np.log10(1 + (1 / df1.d)), 3))
print(df1)

plt.bar(df1['d'], df1['benford'], color = 'dodgerblue',
        edgecolor = 'dodgerblue')
plt.plot(df1['d'], df1['benford'], 'r-o', linewidth = 1.5)
for i, benford_value in enumerate(df1['benford']):
    plt.text(i + 1, benford_value, f'{benford_value * 100:.2f}%',
             ha = 'center', va = 'bottom',
             fontweight = 'bold', color = 'black')
plt.title('Perfect Benford distribution', fontweight = 'bold')
plt.xlabel('First Digit')
plt.ylabel('Distribution Percentage')
plt.xticks(range(1, 9))
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x * 100:.0f}%')
)
plt.show()

print(sum(df1.benford))

print(round(np.log10(2), 3))
print(round(np.log10(1.1111111), 3))

df2 = pd.DataFrame({'row_number': list(range(1, 1001))})

df2 = df2.assign(uniform_distribution =
                 np.random.randint(1, 10, size = 1000))

print(df2.head(n = 3))
print(df2.tail(n = 3))

results1 = df2.groupby('uniform_distribution') \
    .size() \
    .reset_index(name = 'count')
print(results1)

df3 = df2[['row_number']].copy()

df3 = df3.assign(random_distribution =
                 np.random.choice(np.arange(1, 10), size = 1000))

print(df3.head(n = 3))
print(df3.tail(n = 3))

results2 = (df3.groupby('random_distribution') \
            .size() \
            .reset_index(name = 'count'))
print(results2)

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)

ax1.bar(results1['uniform_distribution'], results1['count'],
        color = 'steelblue', edgecolor = 'steelblue')
for i, n in enumerate(results1['count']):
    ax1.text(results1['uniform_distribution'][i], n, str(n),
            ha = 'center', va = 'bottom',
            fontweight = 'bold', color = 'black')
ax1.set_title('Uniform distribution\nn = 1,000', fontweight = 'bold')
ax1.set_xlabel('First Digit')
ax1.set_ylabel('Count')
ax1.set_xticks(range(1, 10))

ax2.bar(results2['random_distribution'], results2['count'],
        color = 'steelblue', edgecolor = 'steelblue')
for i, n in enumerate(results2['count']):
    ax2.text(results2['random_distribution'][i], n, str(n),
            ha = 'center', va = 'bottom',
            fontweight = 'bold', color = 'black')
ax2.set_title('Random distribution\nn = 1,000', fontweight = 'bold')
ax2.set_xlabel('First Digit')
ax2.set_ylabel('Count')
ax2.set_xticks(range(1, 10))

plt.tight_layout()
plt.show()

street_addresses = pd.read_csv('/Users/garysutton/Library/Mobile Documents/com~apple~CloudDocs/Data Sets/street_address_listing.csv',
                  usecols = [0])

print(street_addresses.info())

street_addresses['first_digit'] = (street_addresses['ADDRESS_NO'] \
                                   .apply(lambda x: str(x)[0]))
print(street_addresses.head(10))

unique_values = set(street_addresses.first_digit)
print(unique_values)

street_addresses = street_addresses[street_addresses['first_digit'] \
    .isin(['1', '2', '3', '4', '5', '6', '7', '8', '9'])]

street_addresses['first_digit'] = (street_addresses['first_digit'] \
                                   .astype(int))

bf_street_addresses = bf.first_digits(street_addresses.first_digit,
                                           digs = 1)

populations = pd.read_csv(
    '/Users/garysutton/Library/Mobile Documents/com~apple~CloudDocs/Data Sets/population_by_country_2020.csv',
                  usecols = [1])
print(populations.info())

populations['first_digit'] = (populations['Population'] \
                                   .apply(lambda x: str(x)[0]))
print(populations.head(10))

populations['first_digit'] = populations['first_digit'].astype(int)

bf_populations = bf.first_digits(populations.first_digit, digs = 1)

payments = pd.read_csv(
    '/Users/garysutton/Library/Mobile Documents/com~apple~CloudDocs/Data Sets/corporate_payments.csv',
                  usecols = [3])
print(payments.info())

payments['first_digit'] = (payments['Amount'] \
                                   .apply(lambda x: str(x)[0]))
print(payments.head(10))

payments = payments[(payments.first_digit >= '1') &
                    (payments.first_digit <= '9')]

payments['first_digit'] = payments['first_digit'].astype(int)

bf_payments = bf.first_digits(payments.first_digit, digs = 1)
print(bf_payments)

bf_populations = \
    bf_populations.assign(Expected_Counts = \
                          bf_populations.Expected * \
                          sum(bf_populations.Counts))
print(bf_populations)

x2 = stats.chisquare(bf_populations.Counts,
                     bf_populations.Expected_Counts)
print(x2)

MAD = bf.mad(populations.first_digit, test = 1, decimals = 0)
print(MAD)

AM = populations['first_digit'].mean()
print(AM)

EM = ((1 * 70.742) + (2 * 41.381) + (3 * 29.361) +
     (4 * 22.774) + (5 * 18.608) + (6 * 15.732) +
     (7 * 13.628) + (8 * 12.021) + (9 * 10.753)) / 235
print(EM)

DF = (100 * (AM - EM)) / EM
print(DF)

SD = np.std(populations['first_digit'])
print(SD)

Z = DF / SD
print(Z)

populations['log'] = np.log10(populations['Population'])

populations['mantissa'] = \
        populations['log'] - populations['log'].astype(int)

print(populations.head(10))

print(populations['mantissa'].mean())

print(populations['mantissa'].var())

print(kurtosis(populations['mantissa'], fisher = True))

print(skew(populations['mantissa']))

populations = populations.sort_values(by = 'mantissa')

populations['rank'] = list(range(1, 236))

print(populations.head(3))
print(populations.tail(3))

plt.plot(populations['rank'], populations['mantissa'],
        color = 'slateblue', linewidth = 1.5)
plt.plot([0, 1], [0, 1], transform = ax.transAxes,
        color = 'red', linestyle = '--')
plt.title('Rank Order of Mantissas', fontweight = 'bold')
plt.xlabel('Rank')
plt.ylabel('Mantissa')
plt.show()
