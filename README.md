# Task-2
Perform EDA on Titanic Dataset

1)Load the Dataset
First, you'll need to import the necessary libraries and load the dataset. Here’s how you can do it using Python with pandas and numpy.
import pandas as pd
import numpy as np

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

2)2. Initial Data Inspection
Start by inspecting the first few rows and summary statistics of the dataset to understand its structure and content.
# Display the first few rows of the dataset
print(train_data.head())

# Summary statistics
print(train_data.describe(include='all'))

# Check for missing values
print(train_data.isnull().sum())

3)3. Data Cleaning
Handling Missing Values
Age: Fill missing values with the median or mean, or use more advanced imputation techniques.
Embarked: Fill missing values with the most common port.
Fare: Check for any missing values and fill them if necessary.

# Fill missing Age values with median age
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Fill missing Embarked values with 'S' (most frequent value)
train_data['Embarked'].fillna('S', inplace=True)

# For Fare, fill missing values with the median fare
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

4)Encoding Categorical Variables
Convert categorical variables into numerical values.
# Convert Sex to numeric
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numeric
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

5)4. Exploratory Data Analysis (EDA)
Summary Statistics
Check the distribution of numerical variables and class distributions.

# Summary statistics
print(train_data[['Age', 'Fare', 'Pclass']].describe())

# Check the distribution of classes
print(train_data['Survived'].value_counts(normalize=True))

6)Visualizations
Use visualization libraries like matplotlib and seaborn to explore relationships and patterns.
import seaborn as sns
import matplotlib.pyplot as plt

# Survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Sex')
plt.show()

# Survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Pclass')
plt.show()

# Age distribution
sns.histplot(train_data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Fare distribution
sns.histplot(train_data['Fare'], kde=True)
plt.title('Fare Distribution')
plt.show()

# Correlation heatmap
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

7) Feature Engineering
Create new features or modify existing ones to enhance the model’s performance.
# Create a new feature for family size
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

# Create a feature indicating whether the passenger is alone
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

8) Analyze Relationships
Look for interesting relationships between features and the target variable (Survived).
code:
# Survival rate by Family Size
sns.barplot(x='FamilySize', y='Survived', data=train_data)
plt.title('Survival Rate by Family Size')
plt.show()

# Survival rate by IsAlone
sns.barplot(x='IsAlone', y='Survived', data=train_data)
plt.title('Survival Rate by IsAlone')
plt.show()

9) Model Preparation
Prepare the data for modeling by splitting into features and target variable, and then into training and validation sets.
from sklearn.model_selection import train_test_split

# Features and target
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone']]
y = train_data['Survived']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



