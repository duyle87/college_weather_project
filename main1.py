# project modules
from  collections import Counter
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.metrics import confusion_matrix

matplotlib.use('TkAgg')
df = pd.read_csv('data.csv')
df.drop(['DATE_FOR', 'RTD_ST_CD'], axis=1, inplace=True)
column_name = df.columns


categorical = [var for var in column_name if df[var].dtype == "O"]
numerical = [var for var in column_name if df[var].dtype != "O"]

def plot_missig_cols():  # shows a heatmap with missing cols
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()
plt.figure(figsize=(20,8))
sns.countplot(x=df.Call_Flag, data=df)
plt.show()


# data visualization
corr_matrix = df.corr()
#print(corr_matrix)
plt.figure(figsize=(20,8))
sns.heatmap(corr_matrix, annot=True)
#plt.show()

# shows class distribution with boxpolot
def customer_check(columns):
    check = columns[0]
    if pd.isnull(check) or check == "NONE":
        return 0
    else:
        return int(check)


def c_null(columns):
    check = columns[0]
    if pd.isnull(check):
        return columns.mean()
    else:
        return check

# encoding Cabin cols to handle missing cols
df['CustomerSegment'] = df[['CustomerSegment']].apply(customer_check, axis=1)

df[['M', 'F']] = pd.get_dummies(df['GENDER'])
df.drop(['GENDER', 'MART_STATUS'], axis=1, inplace=True)
print(df.info())
# quit()
# round age columns
df['Age'] = round(df['Age'])

print(df[numerical].isnull().sum())

# missing_col.remove('RECENT_PAYMENT')

cols = ['CHANNEL1_6M', 'CHANNEL2_6M', 'CHANNEL3_6M', 'CHANNEL4_6M', 'CHANNEL5_6M',
        'METHOD1_6M', 'RECENT_PAYMENT', 'PAYMENTS_6M']

for col in numerical:
    col_median = df[col].median()
    df[col].fillna(col_median, inplace=True)
print(df[cols].isnull().sum())
print(df.info())

# standardize Tenure columns
df_x = df[['Age', 'Tenure']]
df[['Age', 'Tenure']] = (df_x - df_x.mean()) / df_x.std()

print(df[['Age', 'Tenure']].head())

X, y = df.drop('Call_Flag', axis=1),  df['Call_Flag']
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

plt.figure(figsize=(20,8))
sns.countplot(x=y_res, data=df)
plt.show()

#Splitting train and test data
print(Counter(df['Call_Flag']))
X_train, X_test, y_train, y_test = \
    train_test_split(X_res,
                    y_res, test_size=0.2,
                     random_state=100)


cols = X_train.columns
print(cols)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

X_train.info(), X_test.info()

# Logistic Regression
logreg = LogisticRegression(solver='liblinear', random_state=0)
logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test)
y_pred_test

y_pred_train = logreg.predict(X_train)
y_pred_train

print(logreg.predict_proba(X_test)[:, 0])
print(logreg.predict_proba(X_test)[:, 1])

from sklearn.preprocessing import binarize
for i in range(4, 7):
    cm1 = 0
    y_pred1 = logreg.predict_proba(X_test)[:, 1]
    y_pred1 = y_pred1.reshape(-1, 1)
    y_pred2 = binarize(y_pred1, threshold=(i / 10))
    cm1 = confusion_matrix(y_test, y_pred2)
    print('With', i / 10, 'threshold the Confusion Matrix is ', '\n\n', cm1, '\n\n',
        'with', cm1[0, 0] + cm1[1, 1], 'correct predictions, ', '\n\n', cm1[0, 1],
        'Type I errors ( Fasle Positive), ', '\n\n',
        cm1[1, 0], 'Type II errors ( False Negative), ', '\n\n',
        'Accuracy score: ', (accuracy_score(y_test, y_pred2)),
        '\n\n',
        'Sensitivity: ', cm1[1, 1] / float(cm1[1, 1] + cm1[1, 0]), '\n\n',
        'Specificity: ', cm1[0, 0] / float(cm1[0, 0] + cm1[0, 1]), '\n\n',
        '==================================================', '\n\n')

#Determining Accuracy Score
print('Train model accuracy score:{0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))
print('Test model accuracy score:{0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n', cm)
print('True Positives (TP) = ', cm[0, 0])
print('True Negatives (TN) = ', cm[1, 1])
print('False Positives (FP) = ', cm[0, 1])
print('False Negatives (FN) = ', cm[1, 0])

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'], index=['Predict Positive', 'Predict Negative'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
#plt.show()

# Determining classification report
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))

#Determine metrics
y_pred1 = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_pred1)

#Plotting the Curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Call_Flag classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()