import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

# Load and investigate the data here:

# Load csv file
df = pd.read_csv('tennis_stats.csv')

# Display first few rows of dataframe
print(df.head())

# Display summary statistics
print(df.describe())

# Display information about the dataframe
print(df.info())

# Perform exploratory analysis here:

# Plot features against outcome variable
sns.pairplot(df, x_vars=df.columns[:-1], y_vars='Winnings', height=5, aspect=0.7)
plt.show()

# Remove non-numerical columns from the dataframe for correlation analysis
df_new = df.select_dtypes(include=['float64', 'int64'])

# Plot a correlation matrix heatmap of the variables
correlation_matrix = df_new.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidth=0.5, annot_kws={"size": 8})
plt.title("Correlation Matrix")
plt.show()

# Perform single feature linear regressions here:

# Selecting one feature and one outcome variable
X = df[['Aces']]
y = df['Wins']

# Split the data into training and test sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, train_size=0.8, random_state=42)

# Create Linear Regression model
model1 = LinearRegression()

# Fitting the model
model1.fit(X_train1, y_train1)

# Making predictions
y_pred1 = model1.predict(X_test1)

# Evaluate the model
mse1 = mean_squared_error(y_test1, y_pred1)
r21 = r2_score(y_test1, y_pred1)

print(f'Mean Squared Error: {mse1}')
print(f'R^2 Score: {r21}')

# Plotting the predictions
plt.scatter(y_test1, y_pred1)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual vs Predicted Wins')
plt.show()

# Selecting a different feature and a different outcome variable
X = df[['BreakPointsFaced']]
y = df['Losses']

# Split the data into training and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, train_size=0.8, random_state=23)

# Create Linear Regression model
model2 = LinearRegression()

# Fitting the model
model2.fit(X_train2, y_train2)

# Making predictions
y_pred2 = model2.predict(X_test2)

# Evaluate the model
mse2 = mean_squared_error(y_test2, y_pred2)
r22 = r2_score(y_test2, y_pred2)

# Print out the score results
print(f'Mean Squared Error: {mse2}')
print(f'R^2 Score: {r22}')

# Plotting the predictions
plt.scatter(y_test2, y_pred2)
plt.xlabel('Actual Losses')
plt.ylabel('Predicted Losses')
plt.title('Actual vs Predicted Losses')
plt.show()

# Perform two feature linear regressions here:

# Selecting two features from data against one predictor variable
X = df[['Aces', 'FirstServe']]
y = df['Wins']

# Split the data into training and test sets
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, train_size=0.8, random_state=10)

# Select the model
model3 = LinearRegression()

# Fit the model
model3.fit(X_train3, y_train3)

# Make predictions
y_pred3 = model3.predict(X_test3)

# Evaluate the model
mse3 = mean_squared_error(y_test3, y_pred3)
r23 = r2_score(y_test3, y_pred3)

# Print out the score results
print(f'Mean Squared Error: {mse3}')
print(f'R^2 Score: {r23}')

# Plotting the predictions
plt.scatter(y_test3, y_pred3)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual vs Predicted Wins')
plt.show()

# Perform multiple feature linear regressions

# Create feature variables to predict against winning outcomes
features = ['Aces', 'FirstServe', 'DoubleFaults', 'BreakPointsFaced']  # Add other relevant features as needed
X = df_new[features]
y = df_new['Winnings']

# Split the data into training and test sets
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, train_size=0.8, random_state=9)

# Select the model
model4 = LinearRegression()

# Fit the model
model4.fit(X_train4, y_train4)

# Make predictions
y_pred4 = model4.predict(X_test4)

# Evaluate the model
mse4 = mean_squared_error(y_test4, y_pred4)
r24 = r2_score(y_test4, y_pred4)

# Print the score results
print('All features:')
print(f'Mean Squared Error: {mse4}')
print(f'R^2 Score: {r24}')

# Feature selection using RFE
selector = RFE(model4, n_features_to_select=5)

# Fit the selector to the training data
selector = selector.fit(X_train4, y_train4)

# Select features for evaluation
selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]
print(f'Selected Features: {selected_features}')

# Evaluate the model with only the selected features
X_train_selected = X_train4[selected_features]
X_test_selected = X_test4[selected_features]

# Fit the model with the selected features
model4.fit(X_train_selected, y_train4)

# Create predictions
y_pred5 = model4.predict(X_test_selected)

# Evaluate score results
mse5 = mean_squared_error(y_test4, y_pred5)
r25 = r2_score(y_test4, y_pred5)

# Print out result scores
print('Selected features model:')
print(f'Mean Squared Error: {mse5}')
print(f'R^2 Score: {r25}')
