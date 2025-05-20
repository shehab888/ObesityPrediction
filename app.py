

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report



# %matplotlib inline

data = pd.read_csv("train_dataset.csv")
testing_data = pd.read_csv("test_dataset.csv")

data.head()

data.info()
testing_data.info()

print(sum(data.duplicated()))
print(sum(testing_data.duplicated()))

#delete dublicated rows
data = data.drop_duplicates()

data.info()

data.isnull().sum()

data.head()

#                                                     fill nulls

data['FCVC'] = data['FCVC'].fillna(data['FCVC'].median())
data['CALC'] = data['CALC'].fillna(data['CALC'].mode()[0])

data.info()

# show the value count for every column

for x in data.columns:
  if data[x].dtype == 'object':
    print(data[x].value_counts(), "\n")

#                                                     handling the outliers

# Get all BoxPlots of the Training set( seaborn and matplotlib )
plt.figure(figsize=(12, 6))
plt.title("Box plots for the Training Set (numerical)")
sns.boxplot(data=data.select_dtypes(include='number'))
plt.xticks(rotation=45)
plt.show()

# Get all BoxPlots of the TestSet using (pandas and matplotlib)
testing_data.select_dtypes(include='number').plot(kind='box', figsize=(12, 6))
plt.title("Box Plot for Test Set (Numerical)")
plt.xticks(rotation=45)
plt.show()

# we can use (IQR , ZScore)
# Outliers function of number columns  using the IQR
def handleOutliers(train):
  # Get the cols dtype (numbers)
  cols=train.select_dtypes(include=np.number).columns
  for col in cols:
    # this is using the quantiles but we can use percentiles(5 and 95 with the data.clip(lower=train[col].quantile(0.05) , upper=train[col].quantile(0.95)))
    Q1=train[col].quantile(0.25)
    Q3=train[col].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-IQR*1.5
    upper_bound=Q3+IQR*1.5
    train[col]=train[col].apply(lambda x: lower_bound if x<lower_bound else (upper_bound if x>upper_bound else x)) # lambda and ternary condition
  return train

# calling the outlier function
data=handleOutliers(data)
testing_data=handleOutliers(testing_data)

# Get the box plots of the x_training set again to check
plt.figure(figsize=(12,6))
plt.title("Box plots for the Training Set (numerical)")
sns.boxplot(data=data.select_dtypes(include='number'))
plt.xticks(rotation=45)
plt.show()

# Get the box plots of x_test set again to check
plt.figure(figsize=(12,6))
plt.title("Box plots for the Test Set (numerical)")
sns.boxplot(data=testing_data.select_dtypes(include='number'))
plt.xticks(rotation=45)
plt.show()

data.info()

#                                                           Encoding the Data

# encodin using label encoder (built in function)
# from sklearn.preprocessing import LabelEncoder

# labelEncoder = LabelEncoder()
# # to can revers the encoding
# inverseEncoder = {}
# inverseEncoderTest = {}

# # loop on the train data columns
# for col in data.columns :
#   if(data[col].dtype=='object'):
#     data[col] = labelEncoder.fit_transform(data[col])
#     inverseEncoder[col] = labelEncoder.classes_

# # loop on the test data columns
# for col in testing_data.columns :
#   if(testing_data[col].dtype=='object'):
#     testing_data[col] = labelEncoder.fit_transform(testing_data[col])
#     inverseEncoderTest[col] = labelEncoder.classes_

data.info()

data.head()

# convert string to int values (encoding manual)

GenderMap = {
    'Male': 1,
    'Female': 0
}

data['Gender'] = data['Gender'].map(GenderMap)
testing_data['Gender'] = testing_data['Gender'].map(GenderMap)

yes_no_map = {
    "yes": 1,
    "no": 0
}
data['family_history_with_overweight'] = data['family_history_with_overweight'].map(yes_no_map)
testing_data['family_history_with_overweight'] = testing_data['family_history_with_overweight'].map(yes_no_map)

data['FAVC'] = data['FAVC'].map(yes_no_map)
testing_data['FAVC'] = testing_data['FAVC'].map(yes_no_map)

Frequency_map = {
    'no': 0,
    'Sometimes': 3,
    'Frequently': 6,
    'Always': 10
}
data['CAEC'] = data['CAEC'].map(Frequency_map)
testing_data['CAEC'] = testing_data['CAEC'].map(Frequency_map)

data['SMOKE'] = data['SMOKE'].map(yes_no_map)
testing_data['SMOKE'] = testing_data['SMOKE'].map(yes_no_map)

data['SCC'] = data['SCC'].map(yes_no_map)
testing_data['SCC'] = testing_data['SCC'].map(yes_no_map)

data['CALC'] = data['CALC'].map(Frequency_map)
testing_data['CALC'] = testing_data['CALC'].map(Frequency_map)

transportaion_map = {
    'Automobile': 0,
    'Public_Transportation': 1,
    'Motorbike': 2,
    'Bike': 3,
    'Walking': 4
}
data['MTRANS'] = data['MTRANS'].map(transportaion_map)
testing_data['MTRANS'] = testing_data['MTRANS'].map(transportaion_map)
# MTRANS_dummies = pd.get_dummies(data['MTRANS'], prefix='MTRANS')
# data = pd.concat([data.drop('MTRANS', axis=1), MTRANS_dummies], axis=1)
# MTRANS_dummies_test = pd.get_dummies(testing_data['MTRANS'], prefix='MTRANS')
# testing_data = pd.concat([testing_data.drop('MTRANS', axis=1), MTRANS_dummies_test], axis=1)



weight_map = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
data['NObeyesdad'] = data['NObeyesdad'].map(weight_map)
testing_data['NObeyesdad'] = testing_data['NObeyesdad'].map(weight_map)

data.info()

data.head()

X = data.drop(columns=['NObeyesdad'])
Y = data['NObeyesdad']

X_Test = testing_data.drop(columns=["NObeyesdad"])
Y_Test = testing_data['NObeyesdad']



#                                                      Normalizing The Data

#Scalling and Normalizing/Standard => we can use built in methods like (MinMaxScaller - StandardScaller) or you can implement with max ,min or std
X = X.drop(columns=["SMOKE"])
X_Test = X_Test.drop(columns=["SMOKE"])

from sklearn.preprocessing import MinMaxScaler
import joblib
scaler=MinMaxScaler()
X_scaler=scaler.fit_transform(X)
joblib.dump(scaler,'scaler.pkl')
X_Test_scaler=scaler.transform(X_Test)

X=pd.DataFrame(X_scaler,columns=X.columns)
X_Test=pd.DataFrame(X_Test_scaler,columns=X_Test.columns)

X.head() # check the normalize on the train data

X_Test.head()  # check the normalize on the test data

#                                                     Feature Selection

# giving the corelation between the features and the target

correlation_with_target = data.corr()['NObeyesdad'].sort_values(ascending=False)
print(correlation_with_target)

#  SMOKE feature it has no corelation but (GENDER  and NCP) has weak coreltaion

# X = X.drop(columns=["NCP", "SMOKE", "Gender"])
# X_Test = X_Test.drop(columns=["NCP", "SMOKE", "Gender"])

# X = X.drop(columns=["SMOKE"])
# X_Test = X_Test.drop(columns=["SMOKE"])

# calling the outliers function for the x and x_test
X=handleOutliers(X)
X_Test=handleOutliers(X_Test)

# Get the box plots of the x_training set again to check
plt.figure(figsize=(12,6))
plt.title("Box plots for the Training Set (numerical)")
sns.boxplot(data=X.select_dtypes(include='number'))
plt.xticks(rotation=45)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calculate correlation matrix
corr = X.corr()

# Create a mask to display only the lower triangle
sns.set(style="white")
mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask

# Set the figure size (adjust depending on number of features)
plt.figure(figsize=(12, 10))

# Create the heatmap
sns.heatmap(
    corr,
    mask=mask,
    annot=True,             # show correlation values
    cmap='coolwarm',        # color map
    fmt=".2f",              # format for numbers
    annot_kws={"size": 9},  # font size of annotations
    linewidths=.5,          # line width between cells
    cbar_kws={"shrink": .6},# shrink colorbar
    square=True             # make cells square
)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Add a title
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')

# Fit layout
plt.tight_layout()

# Show plot
plt.show()

# Get the box plots of x_test set again to check
plt.figure(figsize=(12,6))
plt.title("Box plots for the Test Set (numerical)")
sns.boxplot(data=X_Test.select_dtypes(include='number'))
plt.xticks(rotation=45)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calculate correlation matrix
corr = X_Test.corr()

# Create a mask to display only the lower triangle
sns.set(style="white")
mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask

# Set the figure size (adjust depending on number of features)
plt.figure(figsize=(12, 10))

# Create the heatmap
sns.heatmap(
    corr,
    mask=mask,
    annot=True,             # show correlation values
    cmap='coolwarm',        # color map
    fmt=".2f",              # format for numbers
    annot_kws={"size": 9},  # font size of annotations
    linewidths=.5,          # line width between cells
    cbar_kws={"shrink": .6},# shrink colorbar
    square=True             # make cells square
)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Add a title
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')

# Fit layout
plt.tight_layout()

# Show plot
plt.show()

# KNN_Modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Define the base model
knn = KNeighborsClassifier()

# Define hyperparameters to search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Apply Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Train the model on training data
grid_search.fit(X, Y)

# Get best model
best_knn = grid_search.best_estimator_

# Predict on test data
y_pred = best_knn.predict(X_Test)

# Evaluate performance
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(Y_Test, y_pred))
print("Classification Report:\n", classification_report(Y_Test, y_pred))

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Initialize the model
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
# Train the model
model.fit(X, Y)

# Predict on test data
y_pred = model.predict(X_Test)

# Evaluation
print("Accuracy:", accuracy_score(Y_Test, y_pred))
print(classification_report(Y_Test, y_pred))

train_preds = model.predict(X)

train_acc = accuracy_score(Y, train_preds)

print("Train Accuracy:", train_acc)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Model
rf_model = RandomForestClassifier(max_depth=20,max_features='sqrt',n_estimators=200)

# Parameter Grid
# rf_params = {
#     'n_estimators': [100,200],
#     'max_depth': [None, 5,10, 20],
#     'max_features': ['sqrt', 'log2']
# }

# Grid Search
# rf_grid = GridSearchCV(
#     rf_model,
#     rf_params,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=1
# )

rf_model.fit(X, Y)
y_pred_rf = rf_model.predict(X_Test)
# y_pred_data = rf_grid.best_estimator_.predict(X)

# Evaluation
# print("Acurracy for training: ", accuracy_score(Y, y_pred_data))
# print("Random Forest Best Params:", rf_grid.best_params_)
print("Accuracy:", accuracy_score(Y_Test, y_pred_rf))
print("Classification Report:\n", classification_report(Y_Test, y_pred_rf))

# Logisti Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 200],
    'solver': ['newton-cg'],
}
# Logistic Regression Model
Logisticmodel=LogisticRegression(solver='newton-cg',penalty='l2',C=1000,random_state=42)
# GridSearchCV
grid_search = GridSearchCV(Logisticmodel, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, Y)
best_params = grid_search.best_params_
# Get the best Params
print(f"Best Hyperparameters: {best_params}")
#fit the model
Logisticmodel.fit(X,Y)
# training accuracy
y_pred=Logisticmodel.predict(X_Test)
y_train_pred=Logisticmodel.predict(X)
print(f"the accuracy score of training set {accuracy_score(Y,y_train_pred)}")
print('#'*50)
print(f"the accuracy score of test set {accuracy_score(Y_Test,y_pred)}")
print(f"{classification_report(Y_Test,y_pred)}")
print(f"the confusion matrix is {confusion_matrix(Y_Test,y_pred)}")

# SVM Model
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print("Training basic SVM model...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X, Y)

# Predict on the test set
svm_preds = svm_model.predict(X_Test)

# Evaluate the basic model
print("\n Basic SVM Results:")
print("Test Accuracy:", accuracy_score(Y_Test, svm_preds))
print("\nConfusion Matrix:\n", confusion_matrix(Y_Test, svm_preds))
print("\nClassification Report:\n", classification_report(Y_Test, svm_preds))

# --------------------------------------
# Hyperparameter Tuning with GridSearchCV
# --------------------------------------
print("\nStarting hyperparameter tuning...")
param_grid = {
    'C': [0.1, 1, 10, 100],          # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  # Kernel coefficient
    'kernel': ['rbf', 'poly', 'sigmoid']  # Kernel type
}

# Create GridSearchCV object
grid = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,                # 5-fold cross-validation
    scoring='accuracy',  # Evaluation metric
    n_jobs=-1,          # Use all available cores
    verbose=1           # Show progress
)

# Perform grid search
grid.fit(X, Y)

# Results
print("\n Tuning Results:")
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)

# Get best estimator
best_svm = grid.best_estimator_

# Predict using the best model
best_svm_preds = best_svm.predict(X_Test)

# Evaluate the tuned model
print("\n Tuned SVM Results:")
print("Test Accuracy:", accuracy_score(Y_Test, best_svm_preds))
print("\nConfusion Matrix:\n", confusion_matrix(Y_Test, best_svm_preds))
print("\nClassification Report:\n", classification_report(Y_Test, best_svm_preds))

# Plot feature importance (for linear kernel)
if best_svm.kernel == 'linear':
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(X.columns)), best_svm.coef_[0], align='center')
    plt.yticks(range(len(X.columns)), X.columns)
    plt.title("Feature Importance (Linear SVM)")
    plt.show()

# %%
# Neural Network Model using TensorFlow/Keras
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#
# # One-hot encode the target variable for the neural network
# # Neural networks typically require the target variable to be one-hot encoded for multi-class classification
# encoder = OneHotEncoder(sparse_output=False)
# Y_encoded = encoder.fit_transform(Y.values.reshape(-1, 1))
# Y_Test_encoded = encoder.transform(Y_Test.values.reshape(-1, 1))
#
# # Define the model architecture
# model = keras.Sequential([
#     keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)), # Input layer
#     keras.layers.Dropout(0.2), # Add dropout for regularization
#     keras.layers.Dense(64, activation='relu'), # Hidden layer
#     keras.layers.Dropout(0.2), # Add dropout
#     keras.layers.Dense(Y_encoded.shape[1], activation='softmax') # Output layer
# ])
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(X, Y_encoded, epochs=50, batch_size=32, validation_split=0.2,verbose=0)
#
# # Evaluate the model on the test data
# loss, accuracy = model.evaluate(X_Test, Y_Test_encoded)
# # print(f"\nTest Loss: {loss:.4f}")
# # print(f"Test Accuracy: {accuracy:.4f}")
#
# # Make predictions on the test data
# y_pred_probs = model.predict(X_Test)
# y_pred = tf.argmax(y_pred_probs, axis=1).numpy()
#
# # Convert the one-hot encoded true labels back to original labels for evaluation metrics
# Y_Test_original = tf.argmax(Y_Test_encoded, axis=1).numpy()
#
# # Print classification report and confusion matrix
# print("\nClassification Report:")
# print(classification_report(Y_Test_original, y_pred))
#
# print("\nConfusion Matrix:")
# print(confusion_matrix(Y_Test_original, y_pred))

import numpy as np

# ---------- One-hot Encoding ----------
def one_hot_encode(y, num_classes):
    m = y.shape[0]
    y_encoded = np.zeros((m, num_classes))
    y_encoded[np.arange(m), y] = 1
    return y_encoded

# ---------- Softmax ----------
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ---------- Initialization ----------
def initialize_weights(n_features, n_classes):
    W = np.zeros((n_features, n_classes))
    b = np.zeros((1, n_classes))
    return W, b

# ---------- Forward ----------
def forward(X, W, b):
    Z = np.dot(X, W) + b
    return softmax(Z)

# ---------- Loss with Regularization ----------
def compute_loss(y_true, y_pred, W, reg_lambda=0.01):
    m = y_true.shape[0]
    data_loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    reg_loss = (reg_lambda / (2 * m)) * np.sum(W * W)
    return data_loss + reg_loss

# ---------- Backward with Regularization ----------
def backward(X, y_true, y_pred, W, reg_lambda=0.01):
    m = X.shape[0]
    dZ = y_pred - y_true
    dW = (np.dot(X.T, dZ) / m) + (reg_lambda / m) * W
    db = np.sum(dZ, axis=0, keepdims=True) / m
    return dW, db

# ---------- Predict ----------
def predict(X, W, b):
    probs = forward(X, W, b)
    return np.argmax(probs, axis=1)

# ---------- Accuracy ----------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# ---------- Final Train Function ----------
def train(X, y, X_test, y_test, num_classes, learning_rate=0.1, epochs=1000, reg_lambda=0.01):
    n_samples, n_features = X.shape
    y_encoded = one_hot_encode(y, num_classes)

    W, b = initialize_weights(n_features, num_classes)

    for epoch in range(epochs):
        # Forward
        y_pred = forward(X, W, b)

        # Loss
        loss = compute_loss(y_encoded, y_pred, W, reg_lambda)

        # Backward
        dW, db = backward(X, y_encoded, y_pred, W, reg_lambda)

        # Update
        W -= learning_rate * dW
        b -= learning_rate * db

        # Evaluation
        if epoch % 100 == 0 or epoch == epochs - 1:
            train_acc = accuracy(y, predict(X, W, b))
            test_acc = accuracy(y_test, predict(X_test, W, b))
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    return W, b

W, b = train(X, Y, X_Test, Y_Test, 7, 5, 5000, 0.01)

#
W, b = train(X, Y, X_Test, Y_Test, 7, 3, 5000, 0.01)

y_train_pred = predict(X, W, b)
y_test_pred = predict(X_Test, W, b)

train_acc = accuracy(Y, y_train_pred)
test_acc = accuracy(Y_Test, y_test_pred)

print(f"\n Final Training Accuracy: {train_acc * 100:.2f}%")
print(f" Final Testing Accuracy: {test_acc * 100:.2f}%")

joblib.dump(rf_model,'model.pkl')