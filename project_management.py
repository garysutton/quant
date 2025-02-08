import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import scipy.stats as stats

# illustrative beta distribution - code not in text
# define the PERT parameters
optimistic_time = 3
most_likely_time = 5
pessimistic_time = 10

# calculate the alpha and beta parameters for the beta distribution
alpha = 1 + 4 * (most_likely_time - optimistic_time) / (pessimistic_time - optimistic_time)
beta_param = 1 + 4 * (pessimistic_time - most_likely_time) / (pessimistic_time - optimistic_time)

# generate x values (activity times) between optimistic and pessimistic times
x = np.linspace(optimistic_time, pessimistic_time, 1000)

# calculate the beta distribution pdf
y = beta.pdf((x - optimistic_time) / (pessimistic_time - optimistic_time), alpha, beta_param) / (pessimistic_time - optimistic_time)

# plot the beta distribution
plt.plot(x, y, label = 'Beta Distribution')
plt.fill_between(x, y, alpha = 0.2)

# add vertical lines for the optimistic, most likely, and pessimistic times
plt.axvline(optimistic_time, color = 'green', linestyle = '--', label = 'Optimistic Time')
plt.axvline(most_likely_time, color = 'blue', linestyle = '--', label = 'Most Likely Time')
plt.axvline(pessimistic_time, color = 'red', linestyle = '--', label = 'Pessimistic Time')

# add labels and title
plt.xlabel('Activity Time')
plt.ylabel('Probability Density')
plt.title('PERT Beta Distribution')
plt.legend(loc = 'upper right', bbox_to_anchor = (0.9, 1))

# show the plot
plt.show()

# data frame - define tasks dependencies, t
plan = pd.DataFrame({
    'Activity': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
    'Description': [
        'Define Project Objectives and Scope',
        'Assemble Project Team',
        'Gather Requirements from Stakeholders',
        'Data Collection and Integration',
        'Data Cleaning and Preprocessing',
        'Design Report Templates',
        'Develop Data Processing Pipelines',
        'Implement Report Generation Logic',
        'Develop User Interface for Report Access',
        'Integrate Data Pipelines with Report Generation',
        'Conduct User Testing and Feedback',
        'Finalize and Deploy Automated Reporting Tool'
    ],
    'Dependencies': [None, ['A'], ['A', 'B'], ['C'], ['D'], ['C'], ['E'], ['F'], ['F'], ['G', 'H'], ['I', 'J'], ['K']],
    't': [3, 2, 3, 5, 3, 4, 4, 3, 3, 4, 3, 2]
})

# display
print(plan)

# error handling functions
def errorActivityMsg():
    print('Error in input file: Activity')
    sys.exit(1)

def errorDependenciesMsg():
    print('Error in input file: Dependencies')
    sys.exit(1)

def errortMsg():
    print('Error in input file: t')
    sys.exit(1)

# function to get task index by Activity
def getTaskCode(mydata, code):
    x = 0
    flag = 0
    for i in mydata['Activity']:
        if i == code:
            flag = 1
            break
        x += 1
    if flag == 1:
        return x
    else:
        errorActivityMsg()

# forward pass function to get ES and LS
def forwardPass(mydata):
    ntask = mydata.shape[0]
    ES = np.zeros(ntask, dtype = np.int32)
    EF = np.zeros(ntask, dtype = np.int32)
    temp = []

    for i in range(ntask):
        if not mydata['Dependencies'][i]:
            ES[i] = 0
            try:
                EF[i] = ES[i] + mydata['t'][i]
            except:
                errortMsg()
        else:
            for j in mydata['Dependencies'][i]:
                index = getTaskCode(mydata, j)
                if index == i:
                    errorDependenciesMsg()
                else:
                    temp.append(EF[index])

            ES[i] = max(temp)
            try:
                EF[i] = ES[i] + mydata['t'][i]
            except:
                errortMsg()

        temp = []

    mydata['ES'] = ES
    mydata['EF'] = EF

    return mydata

# backward pass function to get LS and LF
def backwardPass(mydata):
    ntask = mydata.shape[0]
    temp = []
    LS = np.zeros(ntask, dtype = np.int32)
    LF = np.zeros(ntask, dtype = np.int32)
    Successors = np.empty(ntask, dtype = object)

    for i in range(ntask - 1, -1, -1):
        if mydata['Dependencies'][i]:
            for j in mydata['Dependencies'][i]:
                index = getTaskCode(mydata, j)
                if Successors[index] is not None:
                    Successors[index] += mydata['Activity'][i]
                else:
                    Successors[index] = mydata['Activity'][i]

    mydata['Successors'] = Successors

    max_EF = np.max(mydata['EF'])
    for i in range(ntask - 1, -1, -1):
        if mydata['Successors'][i] is None:
            LF[i] = max_EF
            LS[i] = LF[i] - mydata['t'][i]
        else:
            for j in mydata['Successors'][i]:
                index = getTaskCode(mydata, j)
                temp.append(LS[index])

            LF[i] = min(temp)
            LS[i] = LF[i] - mydata['t'][i]
            temp = []

    mydata['LS'] = LS
    mydata['LF'] = LF

    return mydata

# CPM and Slack functions
def slack(mydata):
    ntask = mydata.shape[0]
    Slack = np.zeros(ntask, dtype = np.int32)
    Critical = np.empty(ntask, dtype = object)

    for i in range(ntask):
        Slack[i] = mydata['LS'][i] - mydata['ES'][i]
        if Slack[i] == 0:
            Critical[i] = 'Yes'
        else:
            Critical[i] = 'No'

    mydata['Slack'] = Slack
    mydata['Critical'] = Critical

    mydata = mydata.reindex(
        columns = ['Activity', 'ES', 'EF', 'LS', 'LF', 'Slack', 'Critical'])
    return mydata

# wrapper function
def computeCPM(mydata):
    mydata = forwardPass(mydata)
    mydata = backwardPass(mydata)
    mydata = slack(mydata)
    return mydata

# print function
def printPlan(mydata):
    pd.set_option('display.max_columns', None)
    print('Automated Reporting Tool: Schedule and Slack Times')
    print("*" * 50)
    print('ES = Earliest Start; EF = Earliest Finish;\nLS = Latest Start; LF = Latest Finish;\nSlack = LS - ES')
    print("*" * 50)
    print(mydata)
    print("*" * 50)

# calculate CPM
plan = computeCPM(plan)

# print the plan
printPlan(plan)

# given values
due_date = 30
mean = 29
standard_deviation = 1.41

# calculate the z-score
z_score = (due_date - mean) / standard_deviation

# calculate the cumulative probability
probability = stats.norm.cdf(z_score)

# print the result
print(f'z-score: {z_score:.3f}')
print(f'Probability of completing the project within '
      f'{due_date} weeks: {probability:.2%}')

# code for normal plot - code not in text
import matplotlib.pyplot as plt
# given values
mu = 29
sigma = 1.41
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# plot the normal distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, label = 'Normal Distribution')
ax.fill_between(x, y, where=(x <= 30), color = 'skyblue', alpha = 0.4, label = 'Probability Area')

# annotations
plt.title('Project Completion Time Probability')
plt.xlabel('Weeks')
plt.ylabel('Probability Density')
plt.axvline(mu, color = 'r', linestyle = 'dashed', linewidth = 1)
plt.text(mu, max(y)*0.6, '29 weeks', rotation = 0, verticalalignment = 'center', color = 'r')
plt.axvline(30, color = 'g', linestyle = 'dashed', linewidth = 1)
plt.text(30, max(y)*0.6, '30 weeks', rotation = 0, verticalalignment = 'center', color = 'g')

# add text box for probability
props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)
textstr = '\n'.join((
    r'$\mu=29$ weeks',
    r'$\sigma=1.41$ weeks',
    r'$P(X \leq 30) = 78.5\%$'))
plt.text(0.05, 0.95, textstr, transform = ax.transAxes, fontsize = 14,
        verticalalignment = 'top', bbox = props)

plt.legend()
plt.show()











































