import math
import pandas as pd
import warnings
from matplotlib import pyplot as plt
import seaborn as sns



# ------------------ Ερώτημα 1: ∆ιερεύνηση δεδομένων ( eda.py ) ------------------
df = pd.read_csv('project2_dataset.csv')


# 1. Πόσες είναι οι εγγραφές του συνόλου δεδομένων;
num_records = df.shape[0]
print(f'The number of records in the dataset is: {num_records}')

# 2. Σε τι ποσοστό από αυτές οι χρήστες αγόρασαν τελικά;
revenue_true = df[df['Revenue'] == True]
num_revenue_true = revenue_true.shape[0]
percentage_revenue_true = (num_revenue_true / num_records) * 100
print(f'The percentage of users who ended up buying is: {percentage_revenue_true:.2f}%')

# 3. Ποια είναι η ευστοχία (accuracy) ενός μοντέλου το οποίο προβλέπει πάντα ότι ο χρήστης δε θα αγοράσει (ανεξαρτήτως των χαρακτηριστικών του);
accuracy_not_buy_model = 100 - percentage_revenue_true
print(f'The accuracy of a model that always predicts the user will not buy is: {accuracy_not_buy_model:.2f}%')



# ------------------ Ερώτημα Extra: Plots ------------------
# Filter out FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])
df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
# Convert boolean values to numeric values in any column
df = df.apply(lambda col: col.astype(int) if col.dtype == bool else col)



# ------- Plot heatmap to show the relation for each variable and the revenue -------
C = df.corr()
#print(C['Revenue'])
plt.figure(figsize=(15, 10))
sns.heatmap(pd.DataFrame(C['Revenue']), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.show()



# ------- Plot histograms for each variable to show its distribution -------
# Calculate the number of rows and columns needed for the subplot arrangement
num_cols = 7  # Adjust this value based on the number of columns you want
num_rows = math.ceil(len(df.columns) / num_cols)  # Ceiling division to ensure all variables are accommodated

# Create a figure and axis object with the calculated number of rows and columns
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15))

# Plot histograms for each numerical variable
for ax, column in zip(axes.flatten(), df.columns):
    sns.histplot(df[column], kde=True, color='skyblue', ax=ax)
    ax.set_title(column, fontsize=11)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Hide any empty subplots
for i in range(len(df.columns), num_rows * num_cols):
    axes.flatten()[i].axis('off')

# Adjust layout
plt.tight_layout(pad=5.0)

# Show the plot
plt.show()


