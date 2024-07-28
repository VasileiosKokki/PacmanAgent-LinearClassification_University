import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix




# ------------------ Ερώτημα 2: prepare_data ------------------
def prepare_data(df, train_size, shuffle, random_state):
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
    # Convert boolean values to numeric values in any column
    df = df.apply(lambda col: col.astype(int) if col.dtype == bool else col)
    y = df['Revenue']
    df = df.drop(columns=['Revenue'])
    X = df



    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)
    return X_train, X_test, y_train, y_test



# ------------------ Ερώτημα 3: Προετοιμασία δεδομένων ------------------
df = pd.read_csv('project2_dataset.csv')
X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.70, shuffle=True, random_state=42)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data only
scaler.fit(X_train)

# Transform the training and test data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------ Ερώτημα 4: Υλοποίηση μοντέλου ------------------
regressor = LogisticRegression(penalty=None, max_iter=1000)

# Train the model on the training data
regressor.fit(X_train_scaled, y_train)

# Predict on the training and test sets
yhat_train = regressor.predict(X_train_scaled)
yhat_test = regressor.predict(X_test_scaled)


# ------------------ Ερώτημα 5: Αξιολόγηση μοντέλου ------------------
# Calculate accuracy on the training set
train_accuracy_score = accuracy_score(y_train, yhat_train)

# Calculate accuracy on the test set
test_accuracy_score = accuracy_score(y_test, yhat_test)

# Print the accuracies
print("Accuracy on training set:", train_accuracy_score)
print("Accuracy on test set:", test_accuracy_score)



# Alternative way
mse_ols_train = mean_squared_error(y_train, yhat_train)
mse_ols_test = mean_squared_error(y_test, yhat_test)

print("Accuracy on training set (alternative):", 1 - mse_ols_train)
print("Accuracy on test set (alternative):", 1 - mse_ols_test)


# Calculate confusion matrix for training set
confusion_train = confusion_matrix(y_train, yhat_train)

# Calculate confusion matrix for test set
confusion_test = confusion_matrix(y_test, yhat_test)

print("Confusion matrix for training set:")
print(confusion_train)

print("Confusion matrix for test set:")
print(confusion_test)



# Alternative way 2
TN_1 = confusion_train[0, 0]
FP_1 = confusion_train[0, 1]
FN_1 = confusion_train[1, 0]
TP_1 = confusion_train[1, 1]

TN_2 = confusion_test[0, 0]
FP_2 = confusion_test[0, 1]
FN_2 = confusion_test[1, 0]
TP_2 = confusion_test[1, 1]

train_accuracy = (TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1)
test_accuracy = (TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2)

print("Accuracy on training set (alternative 2):", train_accuracy)
print("Accuracy on test set (alternative 2):", test_accuracy)



