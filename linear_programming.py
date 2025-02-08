import numpy as np
from scipy.optimize import linprog
from dataclasses import dataclass

#data in the form of a dataclass
@dataclass
class Feature:
    epic: str
    name: str
    cost_1yr: int
    total_cost: int
    value_points: int
    effort_points: int

# Creating a list of Feature instances
features = [
    Feature('Enhance Data Infrastructure',
            'Automated Data Cleaning Pipeline',
            50000, 80000, 16, 13),
    Feature('Enhance Data Infrastructure',
            'Real-time Data Ingestion',
            70000, 120000, 18, 14),
    Feature('Enhance Data Infrastructure',
            'Data Privacy and Compliance Tools',
            80000, 130000, 19, 16),
    Feature('Enhance Data Infrastructure',
            'Real-time Data Monitoring and Alerts',
            60000, 100000, 18, 13),
    Feature('Enhance Data Infrastructure',
            'Integration with External Data Sources',
            50000, 85000, 17, 12),
    Feature('Enhance Data Infrastructure',
            'Feature Engineering Automation',
            55000, 90000, 16, 11),
    Feature('Improve Data Analytics and Reporting',
            'Interactive Data Visualization Dashboard',
            60000, 100000, 20, 12),
    Feature('Improve Data Analytics and Reporting',
            'Predictive Maintenance Model', 55000, 90000, 17, 15),
    Feature('Improve Data Analytics and Reporting',
            'Fraud Detection System', 80000, 130000, 20, 14),
    Feature('Improve Data Analytics and Reporting',
            'Automated Reporting Tool',
            45000, 70000, 14, 9),
    Feature('Improve Data Analytics and Reporting',
            'Supply Chain Optimization Model',
            75000, 125000, 17, 15),
    Feature('Improve Data Analytics and Reporting',
            'Anomaly Detection in Real-time Data',
            60000, 95000, 16, 14),
    Feature('Advance Machine Learning and AI Capabilities',
            'Customer Segmentation Analysis',
            40000, 60000, 15, 10),
    Feature('Advance Machine Learning and AI Capabilities',
            'Personalized Marketing Recommendation Engine',
            65000, 110000, 19, 13),
    Feature('Advance Machine Learning and AI Capabilities',
            'Churn Prediction Model',
            50000, 85000, 18, 11),
    Feature('Advance Machine Learning and AI Capabilities',
            'Text Mining for Sentiment Analysis',
            35000, 55000, 13, 8),
    Feature('Advance Machine Learning and AI Capabilities',
            'Customer Lifetime Value Prediction',
            50000, 80000, 15, 10),
    Feature('Advance Machine Learning and AI Capabilities',
            'Time Series Forecasting',
            55000, 90000, 16, 13),
    Feature('Advance Machine Learning and AI Capabilities',
            'Recommendation System for Cross-selling',
            65000, 110000, 18, 12),
    Feature('Advance Machine Learning and AI Capabilities',
            'Automated ML Model Selection and Tuning',
            70000, 115000, 20, 14)
]

# objective function coefficients (negative for maximization)
c = [-f.value_points / f.effort_points for f in features]
print([round(val, 2) for val in c])

# inequality constraints matrix (A_ub) and vector (b_ub)
A_ub = np.array([
    [f.cost_1yr for f in features],
    [f.total_cost for f in features]
])
b_ub = np.array([700000, 1100000])
print(A_ub)
print(b_ub) # not in text

# Inequality constraints for at least 3 features per epic and at most 4 features per epic
epics = ['Enhance Data Infrastructure',
         'Improve Data Analytics and Reporting',
         'Advance Machine Learning and AI Capabilities']

for epic in epics:
    row_min = [-1 if f.epic == epic else 0 for f in features]
    row_max = [1 if f.epic == epic else 0 for f in features]
    A_ub = np.vstack([A_ub, row_min])
    A_ub = np.vstack([A_ub, row_max])
    b_ub = np.append(b_ub, -3)
    b_ub = np.append(b_ub, 4)

# bounds for decision variables (0 or 1)
x_bounds = [(0, 1) for _ in features]

# solve the linear programming problem
result = linprog(c, A_ub = A_ub, b_ub = b_ub,
                 bounds = x_bounds, method = 'highs')
print(result.success)
print(result.x)
print(sum(result.x))
print(round(result.fun * -1, 2))

# check if the solution was successful
if result.success:
    selected_features = [f for f, \
    x in zip(features, result.x) if x == 1]

    print('Selected Features:')
    for f in selected_features:
        print(f'Epic: {f.epic}, '
              f'Feature: {f.name}, '
              f'Cost in 1 Year: ${f.cost_1yr}, '
              f'Total Cost: ${f.total_cost}, '
              f'Priority Score: {f.value_points / f.effort_points:.2f}')

    total_cost_1yr = sum(f.cost_1yr for f in selected_features)
    total_cost = sum(f.total_cost for f in selected_features)
    print(f'\nTotal Cost in 1 Year: ${total_cost_1yr}')
    print(f'Total Cost: ${total_cost}')
else:
    print('Optimization failed.')













