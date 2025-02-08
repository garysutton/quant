import numpy as np

# market shares example
# initial market shares for period 0 - vector of state probabilities
pi_0 = np.array([0.40, 0.35, 0.20, 0.05])

# matrix of transition probabilities
P = np.array([
    [0.85, 0.06, 0.07, 0.02],
    [0.08, 0.80, 0.10, 0.02],
    [0.10, 0.10, 0.75, 0.05],
    [0.12, 0.10, 0.08, 0.70]
])

# calculate market shares for period 1
pi_1 = np.dot(pi_0, P)
print('Market shares for period 1:')
print(pi_1)

# calculate market shares for period 2 using period 1
pi_2_from_1 = np.dot(pi_1, P)
print('Market shares for period 2 given period 1:')
print(pi_2_from_1)

# calculate the two-step transition probability matrix (P^2)
P2 = np.dot(P, P)
print(P2)

# calculate market shares for period 2 directly from period 0
pi_2_direct = np.dot(pi_0, P2)
print('Market shares for period 2 given period 0:')
print(pi_2_direct)

# equilibrium example 1
# initial vector of state probabilities
pi_0 = np.array([1, 0])

# matrix of transition probabilities
P = np.array([
    [0.90, 0.10],
    [0.20, 0.80]
])

# number of periods to predict
n_periods = 20

# initialize a list to store the state probabilities
state_probabilities = [pi_0]

# predict the states over the next 20 periods
for _ in range(n_periods):
    pi_next = np.dot(state_probabilities[-1], P)
    state_probabilities.append(pi_next)

# print the results
for i, pi in enumerate(state_probabilities):
    print(f'Period {i} = {pi}')

# equilibrium example 2
# initial vector of state probabilities
pi_0 = np.array([1, 0])

# matrix of transition probabilities
P = np.array([
    [0.80, 0.20],
    [0.10, 0.90]
])

# number of periods to predict
n_periods = 20

# initialize a list to store the state probabilities
state_probabilities = [pi_0]

# predict the states over the next 20 periods
for _ in range(n_periods):
    pi_next = np.dot(state_probabilities[-1], P)
    state_probabilities.append(pi_next)

# print the results
for i, pi in enumerate(state_probabilities):
    print(f'Period {i} = {pi}')

# define the transition probability matrix P
P = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0.80, 0, 0.10, 0.10],
    [0.30, 0.10, 0.30, 0.30]
])

# Partition the matrix P into sub-matrices I, A, and B (0 not needed)
I = np.array([
    [1, 0],
    [0, 1]
])

A = np.array([
    [0.80, 0],
    [0.30, 0.10]
])

B = np.array([
    [0.10, 0.10],
    [0.30, 0.30]
])

# compute the fundamental matrix F = (I - B)^-1
I_minus_B = I - B
F = np.linalg.inv(I_minus_B)

# compute the final matrix FA = F * A
FA = np.dot(F, A)

# print the resulting matrix FA
print('The final matrix FA is:')
print(FA)

