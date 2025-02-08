import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import poisson
import math
from scipy.stats import norm
from scipy.stats import uniform
import itertools
import scipy.stats as stats
import pandas as pd

probability_heads = 1 / 2
probability_heads_percent = probability_heads * 100
print(f'The probability of success equals: '
      f'{probability_heads_percent}%')

probability_double_sixes = 1 / 36
probability_double_sixes_percent = probability_double_sixes * 100
print(f'The probability of success equals: '
      f'{probability_double_sixes_percent}%')

probability_face_card = 12 / 52
probability_face_card_percent = probability_face_card * 100
print(f'The probability of success equals: '
      f'{probability_face_card_percent}%')

# "long" way of converting odds to probability
result = math.log(0.3)
print(result)
result = math.exp(-1.20)
print(result)
result = .30 / (.30 + 1)
print(result)

# permutations with replacement
# formula = n**r
# total number of items
n = 5
# number of items to choose (size of each permutation)
r = 3
# compute the number of permutations with replacement
num_permutations = n ** r
print(f'Number of permutations with replacement: {num_permutations}')
# alternative - using pow() method)
num_permutations = pow(n, r)
print('Number of permutations with replacement:', num_permutations)

# permutations without replacement
# given values
n = 5
r = 3
# compute the number of permutations
permutations = math.factorial(n) / math.factorial(n - r)
print(f'Number of permutations without replacement: '
      f'{permutations}')
# Compute the number of permutations using math module
permutations = math.perm(n, r)
print(f'Number of permutations without replacement: '
      f'{permutations}')

# combinations without replacement
n = 5
r = 3
# compute the number of combinations
combinations = (math.factorial(n) / \
                (math.factorial(r) * math.factorial(n - r)))
print(f'Number of combinations without replacement: '
      f'{combinations}')
combinations = math.comb(n, r)
print(f'Number of combinations without replacement: '
      f'{combinations}')

# combinations with replacement
n = 5
r = 3
# compute the number of combinations with replacement
combinations = (math.factorial(n + r - 1) / \
                (math.factorial(r) * math.factorial(n - 1)))
print(f'Number of combinations with replacement: '
      f'{combinations}')
# compute the number of combinations with replacement using Python function
combinations = math.comb(n + r - 1, r)
print(f'Number of combinations with replacement: '
      f'{combinations}')

# define parameters
lowest_value = 2
highest_value = 6
mean = 4
std_dev = 1
num_samples = 30000
# generate samples from the normal distribution within the specified range
samples = np.random.normal(mean, std_dev, num_samples)
samples = samples[(samples >= lowest_value) & (samples <= highest_value)]
# plot the histogram
fig, ax = plt.subplots()
ax.hist(samples, bins = 50, range = (lowest_value, highest_value), \
         density = True, color = 'skyblue', edgecolor = 'black')
# add density curve
x_values = np.linspace(lowest_value, highest_value, 30000)
density_curve = norm.pdf(x_values, mean, std_dev)
ax.plot(x_values, density_curve, color = 'black', lw = 2)
# add labels and title, etc.
ax.set_xlabel('Value')
ax.set_ylabel('Probability Density')
ax.set_title('Normal Distribution (mean=4, std dev=1, range=[2, 6])')
plt.grid()
# show plot
plt.show()

# define parameters for the normal distribution - cdf plot
mean = 4
std_dev = 1
# generate data points for the x-axis
x_values = np.linspace(2, 6, 30000)
# calculate the cumulative density function (CDF) for each x-value
cdf_values = norm.cdf(x_values, mean, std_dev)
# plot the CDF
fig, ax = plt.subplots()
ax.plot(x_values, cdf_values, color = 'blue')
# add labels and title, etc.
ax.set_xlabel('Value')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Cumulative Distribution Function (CDF) of Normal Distribution')
plt.grid()
# show plot
plt.ylim(0, 1.1)
plt.show()

# PMF plot
# define the possible outcomes of rolling two six-sided dice
outcomes = np.arange(2, 13)
# calculate the probability of each outcome
probabilities = [1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36]
# plot the PMF
fig, ax = plt.subplots()
ax.bar(outcomes, probabilities, color = 'skyblue', edgecolor = 'black')
# add labels and title
ax.set_xlabel('Sum of Dice')
ax.set_ylabel('Probability')
ax.set_title('Probability Mass Function for Rolling a Pair of Six-Sided Dice')
# set x-axis ticks and limits
plt.xticks(outcomes)
plt.xlim(1.5, 12.5)
# show plot
plt.grid()
plt.show()

# CDF plot - discrete random variables
# define the possible outcomes of rolling two six-sided dice
outcomes = np.arange(2, 13)
# calculate the cumulative probabilities
probabilities = [1/36, 3/36, 6/36, 10/36, 15/36, 21/36, 26/36, 30/36, 33/36, 35/36, 1]
# plot the CDF
fig, ax = plt.subplots()
ax.step(outcomes, probabilities, where = 'post', color = 'blue')
# add labels and title
ax.set_xlabel('Sum of Dice')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Cumulative Distribution Function for Rolling a Pair of Six-Sided Dice')
# set x-axis ticks and limits
plt.xticks(outcomes)
plt.xlim(1.5, 12.5)
# show plot
plt.grid()
plt.show()

# draw a 2x2 grid of normal distributions
# define parameters
means = [0, 0, 0, 0]  # means
std_devs = [5, 10, 15, 20]  # standard deviations
num_samples = 1000  # number of samples
# create subplots
fig, axs = plt.subplots(2, 2, figsize = (10, 8), sharex = True, sharey = True)
# generate and plot normal distributions
for i in range(len(means)):
    # generate random samples from a normal distribution
    samples = np.random.normal(means[i], std_devs[i], num_samples)
    # calculate histogram
    counts, bins = np.histogram(samples, bins = 30, density = True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # plot histogram
    row = i // 2
    col = i % 2
    axs[row, col].bar(bin_centers, counts, width = bins[1] - bins[0], alpha = 0.6)
    # plot probability density function (PDF)
    x = np.linspace(min(samples), max(samples), 100)
    pdf = (1 / (std_devs[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - means[i]) / std_devs[i]) ** 2)
    axs[row, col].plot(x, pdf, 'k', linewidth = 2)
    # add labels and title
    axs[row, col].set_title(f'Std Dev = {std_devs[i]}')
    axs[row, col].set_xlabel('Value')
    axs[row, col].set_ylabel('Probability Density')
    axs[row, col].grid()
# adjust layout
plt.tight_layout()
# display the plot
plt.show()

# calculate number of occurrences within 1, 2, and 3 standard deviations
# define parameters
mean = 0  # mean
std_dev = 1  # standard deviation of the normal distribution
# generate random samples from a normal distribution
samples = np.random.normal(mean, std_dev, 10000)
# calculate the number of occurrences within 1, 2, and 3 standard deviations
within_1_std = np.sum(np.abs(samples - mean) < std_dev)
within_2_std = np.sum(np.abs(samples - mean) < 2 * std_dev)
within_3_std = np.sum(np.abs(samples - mean) < 3 * std_dev)
# print results
print(f'Percent within 1 standard deviation: {within_1_std / 100}%')
print(f'Percent within 2 standard deviations: {within_2_std / 100}%')
print(f'Percent within 3 standard deviations: {within_3_std / 100}%')

# draw standard normal distribution
# get 10,000 samples
samples = np.random.normal(0, 1, 10000)  # mean = 0, standard deviation = 1, 10,000 samples
# plot histogram
plt.hist(samples, bins = 30, density = True, color = 'skyblue', edgecolor = 'black', alpha = 0.6)
# plot PDF curve
x_values = np.linspace(-4, 4, 1000)  # generate x values for the curve
pdf_values = norm.pdf(x_values, loc = 0, scale = 1)  # calculate PDF values for each x value
plt.plot(x_values, pdf_values, color = 'black', lw = 2)  # plot the PDF curve
# add labels, etc.
plt.title('Standard Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid()
plt.show()

# get PDF from standard normal distribution
# using python as a calculator
# constants
mu = 0
sigma = 1
z = 1
# calculate the standard normal PDF using the formula
pdf_value = ((1 / (sigma * math.sqrt(2 * math.pi))) * \
             math.exp(-((z - mu) ** 2) / (2 * sigma ** 2)))
# print the PDF value
print(f'PDF at {z} = {pdf_value}')
# define the parameters
mu = 0
sigma = 1
z = 1
# calculate and print the pdf
pdf_value = norm.pdf(z, loc = mu, scale = sigma)
print('PDF at', z, '=', pdf_value)

# define the value of z
z = 1
# calculate the probability of z
first_probability = norm.cdf(z)
# print the results
print(f'Probability (area): {first_probability * 100}%')
# define a new value for z
z = 1.5
second_probability = norm.cdf(z)
print('Probability (area):', second_probability * 100,'%')
# get the probability for the range
range_probability = second_probability - first_probability
print(f'Probability (area) between '
      f'1.0 and 1.5: {range_probability * 100}%')

# create two plots using the pmf formula
# define parameters
n = 20  # number of trials
k = np.arange(0, n + 1)  # possible number of successes (0 - 20 inclusive)
p = [0.20, 0.50] # values of p for the binomial distributions
# create a 1 x 2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize = (10, 8), sharey = True)
# iterate over each subplot and plot the corresponding binomial distribution
for i, ax in enumerate(axs.flat):
    # calculate the probability mass function (PMF) for the binomial distribution
    pmf = [(np.math.comb(n, j) * (p[i] ** j) * \
             ((1 - p[i]) ** (n - j))) for j in k]
    # plot the binomial distribution
    ax.bar(k, pmf, color = 'skyblue')
    # plot the probability density function (PDF) curve
    ax.plot(k, pmf, color = 'black', lw = 2)
    ax.set_title(f'Binomial Distribution (p = {p[i]})')
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Probability')
    ax.grid()
# adjust layout and display
plt.tight_layout()
plt.show()

# create two plots using Python method
# define parameters
n = 20  # number of trials
k = np.arange(0, n + 1)  # possible number of successes
p = [0.75, 0.90] # values of p for the binomial distributions
# create a 1 x 2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize = (10, 8), sharey = True)
# iterate over each subplot and plot the corresponding binomial distribution
for i, ax in enumerate(axs.flat):
    # calculate the probability mass function for the binomial distribution
    pmf = binom.pmf(k, n, p[i])
    # plot the binomial distribution
    ax.bar(k, pmf, color = 'skyblue')
    # plot the probability density function (PDF) curve
    ax.plot(k, pmf, color = 'black', lw = 2)
    ax.set_title(f'Binomial Distribution (p = {p[i]})')
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Probability')
    ax.grid()
# adjust layout and display
plt.tight_layout()
plt.show()

# compute the probability by using python as calculator
# example values
n = 20  # number of trials
p = 0.5  # probability of success
k = 7  # number of successes
# calculate binomial coefficient (n choose k)
binomial_coefficient = (math.factorial(n) / \
                        (math.factorial(k) * math.factorial(n - k)))
print(binomial_coefficient)
# calculate probability using the binomial formula
probability = binomial_coefficient * (p ** k) * ((1 - p) ** (n - k))
print("Probability:", probability * 100, '%')

# compute the probability using python method
n = 20  # number of trials
p = 0.2  # probability of success (getting heads)
k = 7  # number of successes (getting exactly 3 heads)
# using the pmf (probability mass function) method of the binom object
probability = binom.pmf(k, n, p)
print(f"Probability: {probability * 100}%")

# create 4 uniform distributions as separate plots
fig, axs = plt.subplots(2, 2, figsize = (10, 8), sharex = True, sharey = True)
# define parameters
low = 0  # lower bound of the distribution (a)
high = 10  # upper bound of the distribution (b)
size = 1000  # Number of samples
# generate random samples and plot each uniform distribution
for ax in axs.flat:
    data = np.random.uniform(low, high, size)
    ax.hist(data, bins = 30, density = True, color = 'skyblue', alpha = 0.6)
    ax.set_title('Uniform Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid()
# adjust layout and display the plots
plt.tight_layout()
plt.show()

# discrete uniform distribution
# define the parameters of the discrete uniform distribution
a = 1  # Lower bound
b = 6  # Upper bound
# calculate the probability for each value - equivalent to 1/n
probability = 1 / (b - a + 1)
# values for x-axis (1 to 6)
x_values = list(range(a, b + 1))
# probability for each value
y_values = [probability] * (b - a + 1)
# plotting the discrete uniform distribution
fig, ax = plt.subplots()
ax.bar(x_values, y_values, color = 'skyblue', edgecolor = 'black')
# adding title and labels
ax.set_title('Discrete Uniform Distribution (a=1, b=6)')
ax.set_xlabel('Value')
ax.set_ylabel('Probability')
# configure the aesthetics
plt.ylim(0, 1)
ax.grid()
# display the plot
plt.show()

# compute probabilities arithmetically
# define the parameters of the discrete uniform distribution
a = 1  # lower bound
b = 6  # upper bound
# calculate the total number of possible outcomes
total_outcomes = b - a + 1
# calculate the probability for each value
pmf_values = [1 / total_outcomes] * total_outcomes
# print the probability mass function
for value, probability in zip(range(a, b + 1), pmf_values):
    print(f'p(x = {value}) = {probability * 100}%')

# compute probabilities using Python method
# define the parameters of the discrete uniform distribution
a = 1  # Lower bound
b = 6  # Upper bound
# create a discrete uniform distribution object
uniform_dist = stats.randint(a, b + 1)
# calculate the probability mass function (PMF) for each value
for value in range(a, b + 1):
    probability = uniform_dist.pmf(value)
    print(f'p(x = {value}) = {probability * 100}%')

# set parameters
a = 0 # lower bound
b = 10 # upper bound
x = 7 # probability of this result
# manual computation
manual_pdf = 1 / (b - a)
# compute PDF using SciPy method
pdf_value = uniform.pdf(x, loc = a, scale = b-a)
# print results
print("Manual Computation - PDF at x =", x, ":", manual_pdf)
print("SciPy Method - PDF at x =", x, ":", pdf_value)

# draw 4 poisson distributions in one plot
# values for lambda
lambdas = [2, 5, 8, 10]
# values for x-axis
x_values = np.arange(0, 20)
# plot each Poisson distribution
fig, ax = plt.subplots()
line_styles = ['-', '--', '-.', ':']
for lam, line_style in zip(lambdas, line_styles):
    pmf_values = poisson.pmf(x_values, lam)
    plt.plot(x_values, pmf_values, label = f'λ = {lam}', linestyle = line_style)
# add labels and title
ax.set_xlabel('Number of Events')
ax.set_ylabel('Probability')
ax.set_title('Poisson Distributions')
plt.legend()
plt.grid()
# show plot
plt.show()

# compute poisson probabilities
# define parameters - 2
lam = 2  # Average rate of occurrence
k = 4   # Number of events
# calculate the probability using the formula
probability_formula = ((np.exp(-lam) * lam ** k) / \
                       math.factorial(k))
print(f'Probability (using PMF formula): '
      f'{probability_formula * 100}%')
# calculate the probability using Python
probability_python = poisson.pmf(k, lam)
print(f'Probability (using Python method): '
      f'{probability_formula * 100}%')

# compute poisson probabilities
# define parameters - 10
lam = 10  # Average rate of occurrence
k = 8   # Number of events
# calculate the probability using the formula
probability_formula = ((np.exp(-lam) * lam ** k) / \
                       math.factorial(k))
print(f'Probability (using PMF formula): '
      f'{probability_formula * 100}%')
# calculate the probability using Python
probability_python = poisson.pmf(k, lam)
print(f'Probability (using Python method): '
      f'{probability_formula * 100}%')

# probability of not getting a 4 on one die roll
probability_not_4 = 5 / 6
# probability of not getting a 4 on both dice rolls
probability_not_4_both = pow(probability_not_4, 2)
print(f'Probability of NOT getting at least one 4: '
      f'{probability_not_4_both * 100}%')

# Total number of possible outcomes when rolling two dice
total_outcomes = 6 * 6

# Number of outcomes where the first die shows 3 or 4
favorable_outcomes_first_die = 2

# Number of outcomes where the second die shows 3 or 4
favorable_outcomes_second_die = 2

# Probability of getting 3 or 4 on both dice
probability_both_3_or_4 = (favorable_outcomes_first_die / 6) * (favorable_outcomes_second_die / 6)

print("Probability of getting 3 or 4 on both dice:", probability_both_3_or_4)

# conditional probability - intuitive
# creating a table with the specified values
test_results = {'Negative': [860, 10],
        'Positive': [90, 40]}
# indexing the rows
symptoms = ['No', 'Yes']
# creating and printing the data frame
contingency_table = pd.DataFrame(test_results, index = symptoms)
print(contingency_table)

# extract value in second row and second column - demonstration purposes
contingency_table.iloc[1, 1]

# calculate positive test result | symptoms
P = (contingency_table.iloc[1, 1] / \
       (contingency_table.iloc[1, 1] + contingency_table.iloc[1, 0]))
print(f'Conditional probability: {P * 100}%')

# calculate positive symptoms | test result
P = (contingency_table.iloc[1, 1] / \
       (contingency_table.iloc[0, 1] + contingency_table.iloc[1, 1]))
print(f'Conditional probability: {P * 100}%')






