import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import math
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

### code not in book ###
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x = np.linspace(-10, 10, 100)
y = sigmoid(x)
plt.figure(figsize = (8, 6))
plt.plot(x, y, label= 'Sigmoid Function', color = 'blue', linewidth = 2)
plt.title('Sigmoid Function', fontsize = 16)
plt.xlabel('x', fontsize = 14)
plt.ylabel('Sigmoid(x)', fontsize = 14)
plt.grid()
plt.legend(loc = 'upper left', fontsize = 12)
plt.show()
########

raisins = pd.read_csv('/Users/garysutton/PycharmProjects/raisin_dataset.csv')

print(raisins.info()) # basic data summary

pd.set_option('display.max_columns', None)
print(raisins.head(n = 3))
print(raisins.tail(n = 3))

class_mapping = {'Kecimen': 0, 'Besni': 1}
raisins['Class'] = raisins['Class'].map(class_mapping)

print(raisins[['Area', 'MajorAxisLength', 'Class']].head(n = 3))
print(raisins[['Area', 'MajorAxisLength', 'Class']].tail(n = 3))

# get basic stats segmented by Class
basic_statistics = (raisins.groupby('Class') \
                    .agg(['min', 'max', 'mean', 'median', 'std', 'var']))
print(basic_statistics.round(2))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), \
      (ax7, ax8, ax9)) = plt.subplots(nrows = 3, \
                                      ncols = 3, figsize = (8, 10))

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
columns = ['Area', 'MajorAxisLength', 'MinorAxisLength', \
           'Eccentricity', 'ConvexArea', 'Extent', 'Perimeter']

for i, column in enumerate(columns):
    for class_label, data in raisins.groupby('Class'):
        (data[column] \
         .hist(alpha = 0.5, label = class_label, ax = axes[i]))
    axes[i].set_title(f'{column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    axes[i].legend(labels = ['Kecimen', 'Besni'])

for ax in [ax8, ax9]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# pair of correlation heat maps; segmented by Class
class_0_data = raisins[raisins['Class'] == 0]
class_1_data = raisins[raisins['Class'] == 1]

correlation_matrix_class_0 = class_0_data.corr()
correlation_matrix_class_1 = class_1_data.corr()

fig, axs = plt.subplots(2, 1, figsize = (8, 10))
sns.heatmap(correlation_matrix_class_0, \
            annot = True, cmap = 'coolwarm', ax = axs[0])
axs[0].set_title('Correlation Heatmap - Kecimen Variety')
sns.heatmap(correlation_matrix_class_1, \
            annot = True, cmap = 'coolwarm', ax = axs[1])
axs[1].set_title('Correlation Heatmap - Besni Variety')
plt.tight_layout()
plt.show()

# regression
log_model = smf.logit('Class ~ Area + '
                  'MajorAxisLength + '
                  'MinorAxisLength + '
                  'Eccentricity + '
                  'ConvexArea + '
                  'Extent + '
                  'Perimeter', data = raisins).fit()
print(log_model.summary())

y = raisins['MinorAxisLength']
x = raisins[['Area', 'MajorAxisLength', 'ConvexArea', 'Extent', \
             'Eccentricity', 'Perimeter']]
x = sm.add_constant(x)

multicollinearity_test = sm.OLS(y, x).fit()
r2 = multicollinearity_test.rsquared
print(r2)

y, X = dmatrices('Class ~ Area + '
                 'MajorAxisLength + '
                 'MinorAxisLength + '
                 'ConvexArea + '
                 'Extent + '
                 'Eccentricity + '
                 'Perimeter',
                 data = raisins, return_type = 'dataframe')

vif_df = pd.DataFrame()
vif_df['variable'] = X.columns

vif_df['VIF'] = [variance_inflation_factor(X.values, i) \
                 for i in range(X.shape[1])]

print(vif_df)

# reduced model
reduced_model = smf.logit('Class ~ MinorAxisLength + Perimeter', \
                          data = raisins).fit()
print(reduced_model.summary())

# subset the raisins data frame just those variables in the reduce model
raisins_subset = raisins[['MinorAxisLength', 'Perimeter', 'Class']]

# create derived variable; requires math
raisins_subset['probability'] = 1 / (1 + 2.72**(-(-11.8048 +
              (-.0244 * raisins_subset['MinorAxisLength']) +
              (.0159 * raisins_subset['Perimeter']))))

print(raisins_subset.head(n = 3))
print(raisins_subset.tail(n = 3))

raisins_subset['prediction'] = (raisins_subset['probability'] \
                                .apply(lambda x: 0 if x < 0.5 else 1))

print(raisins_subset.head(n = 3))
print(raisins_subset.tail(n = 3))

# count: Class = 0 and prediction = 0
count = len(raisins_subset[(raisins_subset['Class'] == 0) &
                           (raisins_subset['prediction'] == 0)])
print(count) #397

# count: Class = 0 and prediction = 1
count = len(raisins_subset[(raisins_subset['Class'] == 0) &
                           (raisins_subset['prediction'] == 1)])
print(count) #53

# count: Class = 1 and prediction = 1
count = len(raisins_subset[(raisins_subset['Class'] == 1) &
                           (raisins_subset['prediction'] == 1)])
print(count) #384

# count: Class = 1 and prediction = 0
count = len(raisins_subset[(raisins_subset['Class'] == 1) &
                           (raisins_subset['prediction'] == 0)])
print(count) #66

# create and print confusion matrix
# upper-left quadrant: 397 - number of correct predictions when Class = 0 (tn)
# upper-right quadrant: 53 - number of incorrect predictions when Class = 0 (fp)
# lower-left quadrant: 66 - number of incorrect predictions when Class = 1 (fn)
# lower-right quadrant: 384 - number of correct predictions when Class = 1 (tp)
confusion_matrix = pd.crosstab(raisins_subset['Class'], \
                               raisins_subset['prediction'])
print(confusion_matrix)

tp = 384
tn = 397
fp = 53
fn = 66
n = 900 # number of records

sensitivity = (tp / (tp + fn)) * 100
print(sensitivity) # true positive rate, or recall

# requires sci-kit learn
tpr = recall_score(raisins_subset['Class'], \
                   raisins_subset['prediction'])
print(tpr * 100)

specificity = (tn / (tn + fp)) * 100
print(specificity) # true negative rate

# requires sci-kit learn
tnr = recall_score(raisins_subset['Class'], \
                   raisins_subset['prediction'], pos_label = 0)
print(tnr * 100)

false_positive_rate = (fp / (fp + tn)) * 100
print(false_positive_rate) # aka false alarm rate

false_negative_rate = (fn / (tp + fn)) * 100
print(false_negative_rate)

accuracy = ((tp + tn) / n) * 100
print(accuracy)

misclassification_rate = ((fp + fn) / n) * 100
print(misclassification_rate)

# AUC; requires ski-kit learn
auc = roc_auc_score(raisins_subset['Class'], \
                    raisins_subset['prediction'])
print(auc)

# ROC curve; requires ski-kit learn
fpr, tpr, thresholds = roc_curve(raisins_subset['Class'], \
                                 raisins_subset['prediction'])
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color = 'blue', lw = 2, label = 'ROC curve')
ax.plot([0, 1], [0, 1], color = 'gray', linestyle = '--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve\nAUC = 0.87')
ax.legend(loc = 'lower right')
plt.grid()
plt.show()

fpr, tpr, thresholds = roc_curve(raisins_subset['Class'], \
                                 raisins_subset['prediction'])
plt.plot(fpr, tpr, color = 'blue', lw = 2, label = 'ROC curve')
plt.plot([0, 1], [0, 1], color = 'gray', linestyle = '--')

plt.ylabel('True Positive Rate')
plt.title('ROC Curve\nAUC = 0.87')
plt.legend(loc = 'lower right')
plt.grid()
plt.show()






