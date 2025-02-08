import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import pandas as pd
import yfinance as yf

# define the lambda for the Poisson distribution
lam = 2

# define the range of k values
k_values = np.arange(0, 9)

# calculate the Poisson probabilities for each k value
probabilities = poisson.pmf(k_values, lam)

# convert probabilities to percentages and round to two decimal places
probabilities_percentage = np.round(probabilities * 100, 2)

# create a table with k values and Poisson probabilities
poisson_table = pd.DataFrame({'k': k_values, \
                              'Probability (%)': probabilities_percentage})
print(poisson_table)

# plot the Poisson distribution
fig, ax = plt.subplots()
bars = ax.bar(k_values, probabilities_percentage,
              tick_label = k_values, alpha = 0.6)
ax.set_xlabel('Number of Occurrences (k)')
ax.set_ylabel('Probability (%)')
ax.set_title('Poisson Distribution (λ=2)')
ax.bar_label(bars,
             labels=[f'{prob}%' for prob in probabilities_percentage],
             padding = 3, fontsize = 10, fontweight = 'bold')
ax.set_ylim(0, probabilities_percentage.max() + 5)
plt.tight_layout()
plt.show()

# calculate the cumulative Poisson probabilities for each k value
cumulative_probabilities = poisson.cdf(k_values, lam)

# convert cumulative probabilities to percentages and round to two decimal places
cumulative_percentage = \
    np.round(cumulative_probabilities * 100, 2)

# create a table with k values, Poisson probabilities, and cumulative probabilities
poisson_table = pd.DataFrame({'k': k_values,
                              'Probability (%)': probabilities_percentage,
                              'Cumulative Probability (%)': cumulative_percentage})
print(poisson_table)

# plot the Poisson distribution and its cumulative distribution
fig, ax1 = plt.subplots()
# bar plot for the Poisson probabilities
ax1.bar(k_values, probabilities,
        alpha = 0.6,
        label = 'Probability (%)')
ax1.set_xlabel('k')
ax1.set_ylabel('Probability (%)')
ax1.tick_params('y')
ax1.set_xticks(k_values)
ax1.set_xticklabels([str(k) for k in k_values])
# line plot for the cumulative Poisson probabilities
ax2 = ax1.twinx()
ax2.plot(k_values, cumulative_probabilities,
         color = 'r', marker = 'o',
         label ='Cumulative Probability (%)')
ax2.set_ylabel('Cumulative Probability (%)', color = 'r')
ax2.tick_params('y', colors = 'r')
# add titles and legend
fig.suptitle('Poisson Distribution (λ=2) and Cumulative Distribution')
ax1.legend(loc = 'center right', bbox_to_anchor = (1.00, 0.8))
ax2.legend(loc = 'center right', bbox_to_anchor = (1.00, 0.7))
# display
plt.show()

# generate the list of random digits (00 to 99, 100 in total)
random_digits = [f'{i:02}' for i in range(0, 100)]
# assign ranges of random digits to each k value based on their probability percentage
random_digits_ranges = []
current_index = 0
total_digits = len(random_digits)

for i, row in poisson_table.iterrows():
    count = max(1, int(row['Probability (%)']))
    end_index = (current_index + count - 1) % total_digits
    if current_index <= end_index:
        start = random_digits[current_index]
        end = random_digits[end_index]
        random_digits_ranges.append(f"{start}-{end}")
    else:
        start = random_digits[current_index]
        end = random_digits[end_index]
        random_digits_ranges.append(f"{start}-99, 00-{end}")
    current_index = (end_index + 1) % total_digits

poisson_table['Random Digits'] = random_digits_ranges
print(poisson_table)

# get 10 random digits
# set the seed for reproducibility
np.random.seed(1)
# generate 10 random digits between 00 and 99
random_digits = np.random.choice(100, 10, replace = False)
# format the digits as two-digit strings
formatted_digits = [f'{digit:02}' for digit in random_digits]
print("Random Digits:", formatted_digits)

# expected value
expected_value = sum(poisson_table['k'] * poisson_table['Probability (%)'])
print(expected_value / 100)

# Monte Carlo simulation with 500 trials
np.random.seed(1)  # Ensure reproducibility
num_trials = 500
# normalize probabilities to ensure they sum to 1
normalized_probabilities = probabilities / probabilities.sum()
# run the Monte Carlo simulation
simulated_values = np.random.choice(k_values,
                                    size = num_trials,
                                    p = normalized_probabilities)
# count the frequency of each value in the simulated results
unique, counts = np.unique(simulated_values, \
                           return_counts = True)
simulated_counts = dict(zip(unique, counts))
# convert counts to percentages
simulated_percentages = \
    {k: (v / num_trials) * 100 for k, \
    v in simulated_counts.items()}
# plot the results of the Monte Carlo simulation as a bar chart
fig, ax = plt.subplots()
bars = ax.bar(simulated_percentages.keys(),
              simulated_percentages.values(),
              alpha = 0.6)
# add percentage labels atop the bars in bold font
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height,
            f'{height:.2f}%', ha = 'center', va = 'bottom',
            fontsize = 10, fontweight = 'bold')
# set y-axis to represent percentages
ax.set_ylabel('Percentage (%)')
ax.set_xlabel('Number of Occurrences (k)')
ax.set_title('Monte Carlo Simulation (500 trials) for Poisson Distribution (λ=2)')
ax.set_xticks(k_values)
plt.tight_layout()
plt.show()


# define ticker symbol
ticker_symbol = 'GM'
# get the data for the ticker symbol
ticker_data = yf.Ticker(ticker_symbol)
# get historical market data with custom parameters
stock_data_23 = ticker_data.history(start = '2023-07-01',
                                 end = '2023-12-31',
                                 interval = '1d')
stock_data_24 = ticker_data.history(start = '2024-01-01',
                                 end = '2024-01-31',
                                 interval = '1d')
# subset to include only the 'Close' column
close_prices_23 = stock_data_23[['Close']]
close_prices_24 = stock_data_24[['Close']]
# reset the index to have 0, 1, 2, 3, etc.
close_prices_23 = close_prices_23.reset_index(drop = True)
close_prices_24 = close_prices_24.reset_index(drop = True)

# get basic statistics
stats_23 = close_prices_23.describe()
print(stats_23)
stats_24 = close_prices_24.describe()
print(stats_24)

# calculate daily log returns
close_prices_23['Log Return'] = \
    np.log(close_prices_23['Close'] /
           close_prices_23['Close'].shift(1))
log_returns = close_prices_23['Log Return'].dropna()

# create a density plot of the log returns - requires seaborn
plt.figure(figsize = (10, 6))
sns.kdeplot(log_returns, shade = True)
plt.title('Density Plot of Log Returns for GM Stock')
plt.xlabel('Log Return')
plt.ylabel('Density')
plt.show()

# calculate mean and standard deviation of log returns
mu = log_returns.mean()
print(mu)
sigma = log_returns.std()
print(sigma)

# set up the Monte Carlo simulation parameters
num_simulations = 500
num_days = 20

# get the last closing price of 2023
last_price_23 = close_prices_23.iloc[-1].values[0]

# generate random scenarios for future stock prices based on 2022 closing prices
simulation_df = pd.DataFrame()

for i in range(num_simulations):
    # generate random daily returns
    sampled_returns = np.random.normal(mu, sigma, num_days)
    # simulate stock prices
    price_list = [last_price_23]
    for r in sampled_returns:
        price_list.append(price_list[-1] * np.exp(r))
    simulation_df[i] = price_list

# plot the simulation results
plt.figure(figsize = (10, 6))
plt.plot(simulation_df)
plt.title(f'Monte Carlo Simulation: {ticker_symbol} Closing Stock Price Over {num_days} Days')
plt.xlabel('Trading Days')
plt.ylabel('Closing Price')
plt.show()

# count simulations with ending price greater or less than the starting price
start_price = simulation_df.iloc[0, 0]
ending_prices = simulation_df.iloc[-1, :]
num_greater = (ending_prices > start_price).sum()
num_less = (ending_prices < start_price).sum()
print(num_greater)
print(num_less)

# flatten the simulation results to compute overall statistics
all_simulations = simulation_df.values.flatten()
# get basic statistics across all simulations
overall_stats = pd.DataFrame(all_simulations, \
                    columns = ['Simulated Prices']).describe()
print(overall_stats)































