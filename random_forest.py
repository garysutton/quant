import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import random

# nfl_pbp = pd.read_csv('nfl_pbp.csv')
nfl_pbp = pd.read_csv('/Users/garysutton/PycharmProjects/2023_NFL_PbP_Logs.csv',
                   usecols = ['QUARTER',
                              'DOWN',
                              'TO_GO',
                              'OFFENSIVE_TEAM_VENUE',
                              'SCORE_DIFFERENTIAL',
                              'PLAY_TYPE',
                              'YARDS_GAINED'])

nfl_pbp = nfl_pbp[(nfl_pbp['DOWN'] == 4) & \
                  (nfl_pbp['PLAY_TYPE'].isin(['Pass', 'Run']))]

nfl_pbp['YARDS_GAINED'] = nfl_pbp['YARDS_GAINED'].fillna(0)

nfl_pbp.loc[nfl_pbp['OFFENSIVE_TEAM_VENUE'] == \
            'Road', 'SCORE_DIFFERENTIAL'] *= -1

OFFENSIVE_TEAM_VENUE_mapping = {'Road': 0, 'Home': 1}
nfl_pbp['OFFENSIVE_TEAM_VENUE'] = \
    nfl_pbp['OFFENSIVE_TEAM_VENUE'].map(OFFENSIVE_TEAM_VENUE_mapping)

PLAY_TYPE_mapping = {'Run': 0, 'Pass': 1}
nfl_pbp['PLAY_TYPE'] = \
    nfl_pbp['PLAY_TYPE'].map(PLAY_TYPE_mapping)

nfl_pbp['CONVERT'] = np.where(nfl_pbp['YARDS_GAINED'] < \
                              nfl_pbp['TO_GO'], 0, 1)

print(nfl_pbp.info())

pd.set_option('display.max_columns', None)
print(nfl_pbp.head(10))

# eda
# quarter variable - by convert
# get the counts by quarter and by convert
convert_counts = nfl_pbp['CONVERT'].value_counts()
print(convert_counts)
# prepare data for grouped bar chart
grouped_data = (nfl_pbp.groupby(['QUARTER', 'CONVERT']) \
                .size().unstack(fill_value = 0))
# plot grouped bar chart
grouped_data.plot(kind = 'bar')
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height(),
             str(p.get_height()), ha = 'center', va = 'bottom',
             fontweight = 'bold', color = 'black')
plt.title('Observation Counts by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Count')
plt.xticks(rotation = 0)
plt.legend(title = 'CONVERT', labels = ['No', 'Yes'])
plt.tight_layout()
plt.show()

# to_go - by convert
# get the means and medians
mean_to_go_by_convert = nfl_pbp.groupby('CONVERT')['TO_GO'].mean()
print(mean_to_go_by_convert)
median_to_go_by_convert = nfl_pbp.groupby('CONVERT')['TO_GO'].median()
print(median_to_go_by_convert)
# plot the distributions
# prepare the data
convert_0_data = nfl_pbp[nfl_pbp['CONVERT'] == 0]['TO_GO']
convert_1_data = nfl_pbp[nfl_pbp['CONVERT'] == 1]['TO_GO']
# plot histograms
plt.subplot()
plt.hist(convert_0_data, alpha = 0.5, label = 'CONVERT = No')
plt.hist(convert_1_data, alpha = 0.5, label = 'CONVERT = Yes')
plt.title('Yards Needed for First Down by CONVERT')
plt.xlabel('Yards Needed for First Down')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# score differential - by convert
# get the means and medians
mean_score_differential_by_convert = \
    nfl_pbp.groupby('CONVERT')['SCORE_DIFFERENTIAL'].mean()
print(mean_score_differential_by_convert)
median_score_differential_by_convert = \
    nfl_pbp.groupby('CONVERT')['SCORE_DIFFERENTIAL'].median()
print(median_score_differential_by_convert)
# counts
score_differential_counts = (nfl_pbp['SCORE_DIFFERENTIAL'] \
                   .agg({'negative': lambda x: (x < 0).sum(), \
                         'positive': lambda x: (x > 0).sum(), \
                         'zero': lambda x: (x == 0).sum()}))
print(score_differential_counts)
# plot the distributions
# prepare the data
convert_0_data = \
    nfl_pbp[nfl_pbp['CONVERT'] == 0]['SCORE_DIFFERENTIAL']
convert_1_data = \
    nfl_pbp[nfl_pbp['CONVERT'] == 1]['SCORE_DIFFERENTIAL']
# plot histograms
plt.subplot()
plt.hist(convert_0_data, alpha = 0.5, label = 'CONVERT = No')
plt.hist(convert_1_data, alpha = 0.5, label = 'CONVERT = Yes')
plt.title('Score Differential by CONVERT')
plt.xlabel('Score Differential')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# home / road - by convert
# get the counts by quarter and by convert
# not in book
home_road_counts = \
    nfl_pbp.groupby(['OFFENSIVE_TEAM_VENUE', 'CONVERT']).size().reset_index(name = 'counts')
print(home_road_counts)
# prepare data for grouped bar chart
grouped_data = \
    (nfl_pbp.groupby(['OFFENSIVE_TEAM_VENUE', 'CONVERT']) \
     .size().unstack(fill_value = 0))
# plot grouped bar chart
grouped_data.plot(kind = 'bar')
for p in plt.gca().patches:
    plt.gca().annotate(str(p.get_height()),
                       (p.get_x() + p.get_width() /
                        2., p.get_height()),
                       ha = 'center', va = 'bottom',
                       fontweight = 'bold', color = 'black')
plt.title('Observation Counts by Road vs. Home')
plt.xlabel('Road or Home')
plt.ylabel('Count')
plt.xticks([0, 1], labels = ['Road Team', 'Home Team'],
           rotation = 0)
plt.legend(title = 'CONVERT', labels = ['No', 'Yes'],
           loc = 'upper right')
plt.show()

# play_type - by convert
# prepare data for grouped bar chart
grouped_data = \
    (nfl_pbp.groupby(['PLAY_TYPE', 'CONVERT']) \
     .size().unstack(fill_value = 0))
# plot grouped bar chart
grouped_data.plot(kind = 'bar')
for p in plt.gca().patches:
    plt.gca().annotate(str(p.get_height()),
                       (p.get_x() + p.get_width() /
                        2., p.get_height()),
                       ha = 'center', va = 'bottom',
                       fontweight = 'bold', color = 'black')
plt.title('Observation Counts by Run vs. Pass')
plt.xlabel('Running or Passing Play')
plt.ylabel('Count')
plt.xticks([0, 1], labels = ['Run', 'Pass'],
           rotation = 0)
ax.set_xticklabels(['Run', 'Pass'])
plt.legend(title = 'CONVERT', labels = ['No', 'Yes'])
plt.show()

# get the means and medians
mean_to_go_by_play_type = \
    nfl_pbp.groupby('PLAY_TYPE')['TO_GO'].mean()
print(mean_to_go_by_play_type)
median_to_go_by_play_type = \
    nfl_pbp.groupby('PLAY_TYPE')['TO_GO'].median()
print(median_to_go_by_play_type)

# create and fit the decision tree model
# define X and y; split data into 70/30 train and test
X = nfl_pbp[['QUARTER', 'TO_GO', 'OFFENSIVE_TEAM_VENUE', \
             'SCORE_DIFFERENTIAL', 'PLAY_TYPE']]
y = nfl_pbp['CONVERT']
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.3, random_state = 0)

# verify 70/30 split
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# create the decision tree classifier
clf = DecisionTreeClassifier(criterion = 'gini', \
                             max_depth = 3, random_state = 0)
# train the model
clf = clf.fit(X_train, y_train)
# predict the response on test
y_pred = clf.predict(X_test)

# evaluate the model
clf_accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(f'Accuracy: {round(clf_accuracy, 0)}%')
# confusion matrix
# join y_test and y_pred
results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})
print(results.head(10))
# counts - not in book
# count - both 0 - true negatives
count = len(results[(results['actual'] == 0) &
                            (results['predicted'] == 0)])
print(count)
# count - both 1 - true positives
count = len(results[(results['actual'] == 1) &
                            (results['predicted'] == 1)])
print(count)
# count - actual = 0, predicted = 1 - false positives
count = len(results[(results['actual'] == 0) &
                            (results['predicted'] == 1)])
print(count)
# count - actual = 1, predicted = 0 - false negatives
count = len(results[(results['actual'] == 1) &
                            (results['predicted'] == 0)])
print(count)
# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
# accuracy rate - when 0 - TN / (TN + FP)
accuracy_rate_0 = 71 / (71 + 67) * 100
print(f'Accuracy: {round(accuracy_rate_0, 0)}%')
# accuracy rate - when 1 - TP / (TP + FN)
accuracy_rate_1 = 91 / (91 + 36) * 100
print(f'Accuracy: {round(accuracy_rate_1, 0)}%')

# plot the tree
plt.figure(figsize = (12, 8))
tree.plot_tree(clf, feature_names = ['QUARTER',
                                     'TO_GO',
                                     'OFFENSIVE_TEAM_VENUE',
                                     'SCORE_DIFFERENTIAL',
                                     'PLAY_TYPE'],
               class_names = ['0', '1'],
               filled = True,
               rounded = True,
               fontsize = 12)
plt.tight_layout()
plt.show()

# for gini calculation - play type
train_data = pd.concat([X_train, y_train], axis = 1)
counts = train_data.groupby(['PLAY_TYPE', 'CONVERT']).size()
print(counts)
# for gini calculation - to go
to_go_less_than_3_5 = train_data[train_data['TO_GO'] <= 3.5]
to_go_greater_than_3_5 = train_data[train_data['TO_GO'] > 3.5]
# Group by 'CONVERT' and get counts for TO_GO <= 3.5
counts_less_than_3_5 = (to_go_less_than_3_5 \
                        .groupby('CONVERT').size())
print('Counts for TO_GO <= 3.5:')
print(counts_less_than_3_5)
# Group by 'CONVERT' and get counts for TO_GO > 3.5
counts_greater_than_3_5 = (to_go_greater_than_3_5 \
                           .groupby('CONVERT').size())
print('Counts for TO_GO > 3.5:')
print(counts_greater_than_3_5)

# create and fit the random forest
rf = RandomForestClassifier(n_estimators = 50, \
                            criterion = 'gini', \
                            max_depth = 3, \
                            random_state = 0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(f'Accuracy: {round(rf_accuracy, 0)}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
accuracy_rate_0 = 70 / (70 + 68) * 100
print(f'Accuracy: {round(accuracy_rate_0, 0)}%')
accuracy_rate_1 = 95 / (95 + 32) * 100
print(f'Accuracy: {round(accuracy_rate_1, 0)}%')

# plot feature importance
importances = rf.feature_importances_
features = X.columns

indices = np.argsort(importances)[::-1]
sorted_features = features[indices]
sorted_importances = importances[indices]

plt.figure()
plt.bar(sorted_features, sorted_importances)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Feature Importance in RandomForestClassifier')
plt.tight_layout()
plt.show()

# random trees
random_trees = random.sample(rf.estimators_, 2)
# visualize each of the three trees using matplotlib
for i, tree in enumerate(random_trees):
    plt.figure()
    plot_tree(tree, feature_names = X_train.columns,
              filled = True,
              rounded = True,
              fontsize = 12)
    plt.title(f'Tree {i+1}')
    plt.show()



