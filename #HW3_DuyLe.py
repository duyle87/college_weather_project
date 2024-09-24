import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# import category_encoders as ce
# from distributed import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import torch
from torch import nn
from collections import Counter

# from  string import ascii_letters, punctuation
matplotlib.use('TkAgg')

#Read data set
train = pd.read_csv('train.csv')

#Checking for missing data
def plot_missig_cols():  # shows a heatmap with missing cols
    plt.figure(figsize=(12, 10))
    sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()

print(train.columns)
cols = train.columns
missing_col = [col for col in cols if train[col].isnull().sum() > 0]


# Updating missing rows for Age and Pclass labels
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

train.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
train.dropna(inplace=True)

# Dropping 'Survive' for being key variable
numerical = [col for col in train.columns if train[col].dtype != "O"]
numerical.remove('Survived')

X = pd.concat([train[numerical], pd.get_dummies(train.Embarked), pd.get_dummies(train.Sex), ], axis=1)

y = train.Survived
X.columns = X.columns.astype(str)
smote = RandomOverSampler()
X.info()
y.info()
X_res, y_res = smote.fit_resample(X, y)
nearest_neighbor_removal = Counter(y)[0] - Counter(y_res)[0]
print('Before oversampling %s' % (Counter(y)), 'After oversampling %s' % Counter(y_res), sep='\n')

# X_train = torch.tensor(X_train.values)
# X_test = torch.tensor(X_test.values)
# y_train = y_train.map({'Yes': 1, 'No': 0})
# y_test = y_test.map({'Yes': 1, 'No': 0})
# y_train = torch.tensor(y_train.values)
# y_test = torch.tensor(y_test.values)

# Splitting data to test and train sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=101)

cols = X_train.columns

# Scaling data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  # fit train with scaler
X_test = scaler.fit_transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
# print(len(cols))
# print(X_test)
X_test = pd.DataFrame(X_test, columns=[cols])
# print(X_train.shape)

# Converting data to torch tensors
from torch.utils.data import Dataset, DataLoader
class Input_Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


batch_size = 64
# Instantiating training and test data
train_data = Input_Data(X_train.to_numpy(), y_train.to_numpy())
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = Input_Data(X_test.to_numpy(), y_test.to_numpy())
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

#Building Neural Network
class DoubleHiddenLayer(nn.Module):
    def __init__(self, in_dim, hid_dim1,  hid_dim2, out_dim):
        super(DoubleHiddenLayer, self).__init__()
        # Initiate first hidden layer
        self.hid_lay1 = nn.Linear(in_features=in_dim, out_features=hid_dim1)
        self.relu1 = nn.ReLU()

        # Initiate second hidden layer
        self.hid_lay2 = nn.Linear(in_features=hid_dim1, out_features=hid_dim2)
        self.tanh2 = nn.Tanh()

        self.output = nn.Linear(in_features=hid_dim2, out_features=out_dim)
        self.relu2 = nn.Sigmoid()


    def forward(self, x):
        # Compute the output of the first hidden layer
        x = self.hid_lay1(x)
        x = self.relu1(x)

        # Compute the output of the second hidden layer
        x = self.hid_lay2(x)
        x = self.tanh2(x)
        x = self.output(x)
        x = self.relu2(x)
        return x


# Instantiating the neural network
model = DoubleHiddenLayer(11, 5, 3, 1)
print(model)
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()
epochs = 100
loss_arr = []

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# for i in range(epochs):
#     y_pred = model.forward(X_train.float())
#     loss = criterion(y_pred, y_train)
#     loss_arr.append(loss.detach().numpy())
#
#     if i % 200 == 0:
#         print(f'Epoch: {i} Loss： {loss}')

#Optimizing Model
for epoch in range(epochs):
    for X, y in train_dataloader:
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_arr.append(loss.item())
        loss.backward()
        optimizer.step()
print("Model is optimized")

plt.title('Loss vs Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(loss_arr)
plt.show()

# Checking Accuracy Score
from itertools import chain
y_prediction, y_result = [], []
total, correct = 0, 0
with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        prediction = np.where(outputs > 0.4, 1, 0)
        prediction = list(chain(*prediction))
        y_prediction.append(prediction)
        y_result.append(y)
        total += y.size(0)
        correct += (prediction == y.numpy()).sum().item()
print(f'Accuracy of the neural network model is: {100 * correct // total}%')

y_prediction = list(chain(*y_prediction))
y_result = list(chain(*y_result))

# Final Classification Report
print(classification_report(y_result, y_prediction))
