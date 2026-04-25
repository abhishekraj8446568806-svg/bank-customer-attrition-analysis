## Data Loading and Preprocessing

import pandas as pd

# Load the dataset
df = pd.read_csv("bank_dataset_v2.csv",header=0,skiprows=[1])

# Display the first few rows
print(df.head())

# Print the column names of the dataset.
print(df.columns) 

# Standardize the column names: convert to lowercase and replace spaces with underscores.

# Replacing all spaces with underscore(_) and converting the entire string to lowercase

df.columns = df.columns.str.lower().str.replace(" ", "_")

# Showing the column headers again
df.columns

 ## Handle Missing Values


# Check for missing values
print(df.isnull().sum()) 

# What it does:
# df.isnull() → Returns a DataFrame of the same shape as df, with True where a value is missing (NaN) and False otherwise.
# .sum() → Counts the number of True values (i.e., missing values) in each column.
# print(...) → Displays the result.

# Impute missing 'salary' values with median
df['salary'].fillna(df['salary'].median(), inplace=True)

# Breaking it Down:
# df['Salary'] → Selects the "Salary" column from the DataFrame.
# .fillna(df['Salary'].median(), inplace=True) → Replaces all NaN (missing) values in the "Salary" column with the median of the column.
# df['Salary'].median() → Calculates the median salary (the middle value when sorted).
# fillna(value, inplace=True):
# value: The value used to replace NaNs (here, it's the median salary).
# inplace=True: Updates the DataFrame directly instead of returning a new one.

# Impute missing 'balance' values with 0
df['balance'].fillna(0, inplace=True)

# Impute missing 'satisfaction_score' with median
df['satisfaction_score'].fillna(df['satisfaction_score'].median(), inplace=True)

# Drop rows where 'gender' is missing
df = df.dropna(subset=['gender'])

# Fill missing 'card_type' with the most frequent value (mode)
df['card_type'].fillna(df['card_type'].mode()[0], inplace=True)


# Print the count of missing values in each column
print(df.isnull().sum())  # Count of NaN values

#  Convert 'card_type' to uppercase and Strip whitespace from 'card_type'
df["card_type"] = df["card_type"].str.upper().str.strip()
df['card_type'].unique()

## Explore Numeric Columns

# Identify all numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
print(numeric_columns) 

# Count unique values in each numeric column
 # Step 1: Get the count of unique values for each numeric column
unique_counts = {col: df[col].nunique() for col in numeric_columns}

# Step 2: Print or use the unique counts
print(unique_counts)

##  Outlier Detection and Treatment

import matplotlib.pyplot as plt

# Plot boxplot for 'salary'
# Chart title: Salary Distribution
# Plot Salary distribution
plt.figure(figsize=(8, 4))
df["salary"].plot(kind="box", vert=False)
plt.title("Salary Distribution")
plt.show()

# Plot boxplot for 'balance'
# Chart title: Balance Distribution 
plt.figure(figsize=(8, 4))
df["balance"].plot(kind="box", vert=False)
plt.title("Balance Distribution")
plt.show()

# Identify and count outliers in salary using IQR method
 #  Calculate IQR for Salary
Q1 = df["salary"].quantile(0.25) 
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1

# Define outlier thresholds
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

# Identify outliers
salary_outliers = df[(df["salary"] < lower_bound) | (df["salary"] > upper_bound)]

print("Outlier Customers Based on Salary:")
print(salary_outliers[["salary"]])

outlier_count = ((df["salary"] < lower_bound) | (df["salary"] > upper_bound)).sum()
print("Count of Outlier Customers Based on Salary:", outlier_count)

# Cap salary outliers at the upper bound
import numpy as np
df["salary"] = np.where(df["salary"] > upper_bound, upper_bound, df["salary"])


# Plot Salary distribution after handling outiers
# Chart title: Salary Distribution - Outliers Handled

# Plot Salary distribution
plt.figure(figsize=(8, 4))
df["salary"].plot(kind="box", vert=False)
plt.title("Salary Distribution - Outliers Handled")
plt.show()


## Exploratory Data Analysis

# Calculate Basic Summary Statistics
# Compute the mean and median for key numeric columns: Salary , Balance and Credit Score
# Print the results for interpretation.


mean_salary = round(df["salary"].mean())
median_salary = round(df["salary"].median())

mean_balance = round(df["balance"].mean())
median_balance = round(df["balance"].median())

mean_credit_score = round(df["credit_score"].mean())
median_credit_score = round(df["credit_score"].median())

# Display results
print(f"Mean Salary: {mean_salary}, Median Salary: {median_salary}")
print(f"Mean Balance: {mean_balance}, Median Balance: {median_balance}")
print(f"Mean Credit Score: {mean_credit_score}, Median Credit Score: {median_credit_score}")

# Understand Categorical Variables
# Count and display how many customers fall into each category of: Gender, Card Type, HasLoan, HasFD

# Count of each category
gender_count = df["gender"].value_counts()
card_type_count = df["card_type"].value_counts()
loan_status_count = df["hasloan"].value_counts()
fd_status_count = df["hasfd"].value_counts()

# Display results
print("Gender Distribution:\n", gender_count)
print("\nCard Type Distribution:\n", card_type_count)
print("\nLoan Status Distribution:\n", loan_status_count)
print("\nFixed Deposit Status Distribution:\n", fd_status_count)

# PLot the above count distribution --> Optional 
import matplotlib.pyplot as plt

# Plot gender distribution
plt.figure(figsize=(6, 4))
gender_count.plot(kind='bar')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot card type distribution
plt.figure(figsize=(6, 4))
card_type_count.plot(kind='bar')
plt.title("Card Type Distribution")
plt.xlabel("Card Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot loan status distribution
plt.figure(figsize=(6, 4))
loan_status_count.plot(kind='bar')
plt.title("Loan Status Distribution")
plt.xlabel("Has Loan")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot fixed deposit status distribution
plt.figure(figsize=(6, 4))
fd_status_count.plot(kind='bar')
plt.title("Fixed Deposit Status Distribution")
plt.xlabel("Has FD")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Create a boxplot for the Balance column to detect outliers and understand spread.

import matplotlib.pyplot as plt
import seaborn as sns

# Balance using boxplots
plt.figure(figsize=(10,5))
sns.boxplot(x=df["balance"])
plt.title("Balance Distribution")
plt.xlabel("Balance")
plt.show()


# Select a random sample of 200 customers from the dataset.
# Plot scatter plots to explore how two numeric variables relate to each other, plot for Credit Score vs Balance


# Subset of data (random sampling) around 200 datapoints are taken here
df_sample = df.sample(n=200, random_state=42)

# Scatter plot to detect anomalies in Credit Score vs Balance
plt.figure(figsize=(10,5))
sns.scatterplot(x=df_sample["credit_score"], y=df_sample["balance"])
plt.title("Credit Score vs Balance")
plt.xlabel("Credit Score")
plt.ylabel("Balance")
plt.show()

## Feature Engineering

# Add the following new columns to the dataset:
# Debt-to-Income Ratio
#  Formula: (Balance + (HasLoan × Salary × 0.3)) / Salary

# 1. Financial Stability Indicator
df["Debt-to-Income Ratio"] = (df["balance"] + (df["hasloan"] * df["salary"] * 0.3)) / df["salary"]

print(df[["first_name", "Debt-to-Income Ratio"]].tail())

# Create new column calculating Loyalty Score
# Formula: (Tenure × Satisfaction Score) / (1 + Count of Complains)

df["Loyalty Score"] = (df["tenure"] * df["satisfaction_score"]) / (1 + df["count_of_complains"])

# Display dataset with new features
print(df.head())

## Analyze Complaints by State

# Compute the average number of complaints per state using groupby() and transform().

df["State Avg Complaints"] = df.groupby("state")["count_of_complains"].transform("mean")

# For each customer, compare their complaint count to their state’s average.
# Create a flag called High Complainer:
# 1 if their complaints are above the state average
# 0 otherwise
df["Above State Avg Complaints"] = df["count_of_complains"] > df["State Avg Complaints"]

df["High Complainer"] = df["Above State Avg Complaints"].astype(int)


# Print first 10 rows with updated data to see the difference
df.head(10)

## Univariate Analysis: Categorize customers by Salary

# Create a new column called Salary Category with the following buckets:
# Low (≤ 50,000)
# Medium (50,001 – 100,000)
# High (100,001 – 150,000)
# Very High (150,001 – 200,000)
# Above 2 Lakhs (> 200,000)
# Count how many customers fall into each group and plot the result.


df["Salary Category"] = np.where(df["salary"] <= 50000, "Low",
                         np.where(df["salary"] <= 100000, "Medium",
                         np.where(df["salary"] <= 150000, "High",
                         np.where(df["salary"] <= 200000, "Very High", "Above 2 Lakhs"))))

# Count customers in each category
salary_counts = df["Salary Category"].value_counts()

# series datatype
print(salary_counts)

# Plot salary distribution
# Here, index indicates the labels and values indicate total no of customers that fall into that label
plt.bar(salary_counts.index, salary_counts.values, color="skyblue")
plt.xlabel("Salary Category")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution by Salary Category")
plt.show()


## Compare Customer Segments Using Grouped Statistics

## Bivariate Analysis

# Calculate Average Number of Products Based on Customer Tenure

tenure_product_analysis = df.groupby("tenure")["num_of_products"].mean()
# Display results
print("Average Number of Products Based on Customer Tenure:")
print(tenure_product_analysis)

## Multivariate Analysis

# Grouping by churn status to analyze salary and product usage
churn_analysis = df.groupby("exited")[["salary", "num_of_products"]].mean()
print(churn_analysis)

# Bar chart to compare salary for exited vs. retained customers
plt.figure(figsize=(6, 4))
plt.bar(["stayed", "exited"], churn_analysis["salary"], color=["green", "red"])
plt.title("Average Salary - Stayed vs. Exited")
plt.ylabel("Average Salary")
plt.show()


# Bar chart to compare product usage for exited vs. retained customers
plt.figure(figsize=(6, 4))
plt.bar(["stayed", "exited"], churn_analysis["num_of_products"], color=["green", "red"])
plt.title("Average Number of Products - Stayed vs. Exited")
plt.ylabel("Number of Products")
plt.show()

# Demographic Analysis

Demographic factors are measurable characteristics of a population, typically used to segment and understand customers better.
Without understanding these factors, companies may:
- Offer generic solutions that don’t resonate with specific customer segments.
- Miss out on early signs of churn in high-risk groups.
- Fail to optimize marketing campaigns for the right audiences.


# The dataset got updated and we have a new version though this is in cleaned format already with the missing values handled
import pandas as pd

# Load the dataset
df = pd.read_csv("https://gitlab.crio.do/me_notebook/me_jupyter_bankattritionanalysis/-/raw/master/bank_dataset_v3.csv",header=0,skiprows=[1])

# Display the first few rows
print(df.head())

df["card_type"] = df["card_type"].str.upper().str.strip()
df['card_type'].unique()

### 1. Visualize customers by churn status

# 1. Count the number of churned and non-churned customers using value_counts() on the 'exited' column.
# 2. Create labels for the pie chart, 'Retained' for non-churned and 'Churned' for exited customers.
# 3. Generate a pie chart to visualize the proportion of churned vs. retained customers. 
#    - Display the labels on the pie chart with percentages formatted to 1 decimal place.
#    - Add a title "Customer Churn Proportion".


# Step 1: Count no of churned and non churned customers 
churn_counts = df['exited'].value_counts()
# print(churn_counts)
# Step 2: Label retained and churned 
labels = ['Retained', 'Churned']
# Step 3: Create pie chart
# Create pie chart
plt.figure(figsize=(6, 6))
churn_counts.plot(kind='pie', labels=labels, autopct='%1.1f%%')
plt.title('Customer Churn Proportion')
plt.show()

### 2. Churn variation across states

# 1. Group the data based on the 'state' column using groupby().
# 2. Extract the 'exited' column for churned customers within each state.
# 3. Calculate the total number of churned customers per region by summing the 'exited' column for each group.
# 4. Plot a pie chart to visualize the churn distribution by region. 
#    - Display the percentages of churned customers for each region.
#    - Add a title "Churn Proportion by Region".

# Step 1: Group the data by region
region_grouped = df.groupby('state')
# Step 2: Extract the 'exited' column for churned customers
region_exited = region_grouped['exited']
# Step 3: Calculate the total number of churned customers per region
region_churn_counts = region_exited.sum()
# Step 4: Plot a pie chart to visualize churn distribution by region
plt.figure(figsize=(8, 6))
region_churn_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3')
plt.title('Churn Proportion by State')
plt.show()

### 3. Visualize age vs exit status

# 1. Count the number of churned and non-churned customers by age:
#    - Filter the data for churned customers (exited == 1) and group by 'age', then count the occurrences.
#    - Filter the data for non-churned customers (exited == 0) and group by 'age', then count the occurrences.
# 2. Plot a scatter plot to visualize the number of churned vs non-churned customers across ages:
#    - Use red color for churned customers and blue for non-churned customers.
#    - Set the title to "Scatter Plot: Age vs Number of Customers (Churned vs Non-Churned)".
#    - Label the x-axis as "Age" and the y-axis as "Number of Customers".
#    - Add a legend to differentiate between churned and non-churned customers.

# Step 1: Count churned and non-churned customers by age
churned_count = df[df['exited'] == 1].groupby('age').size()
non_churned_count = df[df['exited'] == 0].groupby('age').size()
# Step 2: Plot in a scatter plot
sns.scatterplot(x=churned_count.index, y=churned_count.values, color='red', label='Churned')
sns.scatterplot(x=non_churned_count.index, y=non_churned_count.values, color='blue', label='Non-Churned')
# Step 3: Customize the plot
plt.title("Scatter Plot: Age vs Number of Customers (Churned vs Non-Churned)")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.legend()
plt.show()


### 4. Visualize age-distribution and exit status

# 1. Create a box plot to visualize the distribution of age by churn status:
#    - Use the 'exited' column for the x-axis (churn status) and the 'age' column for the y-axis (age distribution).
#    - Set the title to "Box Plot: Age Distribution by Churn Status".
#    - Label the x-axis as "Churn Status (Exited)".
#    - Label the y-axis as "Age".


plt.figure(figsize=(10, 6))
sns.boxplot(x='exited', y='age', data=df)
plt.title("Box Plot: Age Distribution by Churn Status")
plt.xlabel("Churn Status (Exited)")
plt.ylabel("Age")
plt.show()


### 5. Visualize gender and exit status

# 1. Filter the dataset to get only customers who have exited (churned) by using the 'exited' column.
# 2. Count the total number of customers per gender by grouping the data by 'gender' and counting the 'exited' column.
# 3. Count the number of churned customers per gender by filtering for churned customers and grouping by 'gender'.
# 4. Calculate the churn rate per gender manually:
#    - Divide the number of churned customers by the total number of customers per gender, then multiply by 100 to get the percentage.
# 5. Plot a bar plot to visualize the churn rate by gender:
#    - Plot the churn rate values against gender on the x-axis.
#    - Set the title to "Churn Rate Distribution by Gender".
#    - Label the x-axis as "Gender" and the y-axis as "Churn Rate (%)".


# Step 1: Filter for customers who have exited (churned)
churned_customers_df = df[df['exited'] == 1]
# Step 2: Count total customers per gender
total_customers_by_gender = df.groupby('gender')['exited'].count()
# Step 3: Count churned customers per gender
churned_customers_by_gender = churned_customers_df.groupby('gender')['exited'].count()
# Step 4: Calculate churn rate manually
churn_rate_by_gender = (churned_customers_by_gender / total_customers_by_gender) * 100
# Step 5: Plot churn rate by gender
plt.figure(figsize=(10, 6))
sns.barplot(x=churn_rate_by_gender.index, y=churn_rate_by_gender.values)
plt.title("Churn Rate Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")
plt.show()


### 6. Churn Rate by Region and Gender

# 1. Group the data by both 'state' and 'gender' using the groupby() function.
# 2. Count the total number of customers in each group by counting the 'exited' column for each group.
# 3. Count the churned customers in each group by summing the 'exited' column (since 'exited' is 1 for churned customers).
# 4. Calculate the churn rate for each group manually:
#    - Divide the number of churned customers by the total number of customers in each group, then multiply by 100 to get the percentage.
# 5. Unstack the churn rate to separate the data by gender for better visualization in the plot.
# 6. Plot a bar chart to visualize the churn rate by state and gender:
#    - Use a bar plot with black edges around the bars.
#    - Set the title to 'Churn Rate by State and Gender'.
#    - Label the x-axis as 'State' and the y-axis as 'Churn Rate (%)'.

# Step 1: Group data by state and gender
grouped_data = df.groupby(['state', 'gender'])

# Step 2: Count total customers for each group
total_customers = grouped_data['exited'].count()

# Step 3: Count churned customers for each group
churned_customers = grouped_data['exited'].sum()  # Since 'exited' is 1 for churned customers

# Step 4: Calculate churn rate manually
churn_rate = (churned_customers / total_customers) * 100

churn_rate_unstacked = churn_rate.unstack()

# Step 5: Plot a bar chart 
plt.figure(figsize=(12, 6))

churn_rate_unstacked.plot(kind='bar', edgecolor='black')

# Step 6: Customize the plot
plt.title('Churn Rate by State and Gender')
plt.xlabel('State')
plt.ylabel('Churn Rate (%)')
plt.show()



### 7. Relationship between income groups - salary and churn rate.

# 1. Create income bins (ranges) to categorize customers by income:
#    - Define the bins as [0, 30000, 50000, 70000, 100000, 150000] to represent different income ranges.
#    - Label the bins as '<30K', '30K-50K', '50K-70K', '70K-100K', '>100K'.
# 2. Assign customers to income groups based on their salary using the pd.cut() function and the defined bins and labels.
# 3. Count the total number of customers in each income group by grouping by 'income_group' and counting the 'exited' column.
# 4. Count the number of churned customers in each income group by grouping by 'income_group' and summing the 'exited' column.
# 5. Calculate the churn rate for each income group manually:
#    - Divide the number of churned customers by the total number of customers in each income group, then multiply by 100 to get the percentage.
# 6. Plot a bar chart to visualize the churn rate by income group:
#    - Use a bar chart to display the churn rate for each income group.
#    - Set the title to 'Churn Rate by Income Group'.
#    - Label the x-axis as 'Income Group' and the y-axis as 'Churn Rate (%)'.


# Step 1: Create income bins (ranges)
bins = [0, 30000, 50000, 70000, 100000, 150000]  # Example income ranges
labels = ['<30K', '30K-50K', '50K-70K', '70K-100K', '>100K']

# Step 2: Assign income group labels
df['income_group'] = pd.cut(df['salary'], bins=bins, labels=labels)

# Step 3: Count total customers in each income group
total_customers_by_income = df.groupby('income_group')['exited'].count()

# Step 4: Count churned customers in each income group
churned_customers_by_income = df.groupby('income_group')['exited'].sum()

# Step 5: Calculate churn rate manually and convert to percentage
churn_rate_by_income = (churned_customers_by_income / total_customers_by_income) * 100

# Step 6: Plot a bar chart
plt.figure(figsize=(10, 6))
churn_rate_by_income.plot(kind='bar')
plt.title('Churn Rate by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Churn Rate (%)')
plt.show()

# 1. Define bins and labels for age groups:
#    - Set the age bins as [18, 30, 45, 60, 100] to categorize customers into different age ranges.
#    - Define the corresponding labels for these age ranges as ['18-30', '31-45', '46-60', '60+'].
# 2. Apply pd.cut() to create age groups in the 'age_group' column:
#    - Use the defined age bins and labels, and set right=False to include the left endpoint of each bin.
# 3. Print the first few rows of the dataframe to verify the new 'age_group' column.


# Step 1: Define bins and labels for age groups
age_bins = [18, 30, 45, 60, 100]  # Age ranges
age_labels = ['18-30', '31-45', '46-60', '60+']  # Corresponding labels

# Step 2: Apply pd.cut() to create age groups
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

print(df.head())

### 8. Churn rates across regions and genders

# 1. Group the data by 'region' and 'gender' using the groupby() function.
# 2. Count the total number of customers in each region-gender group by counting the 'exited' column.
# 3. Count the number of churned customers in each region-gender group by summing the 'exited' column (since 'exited' is 1 for churned customers).
# 4. Calculate the churn rate for each region-gender group manually:
#    - Divide the number of churned customers by the total number of customers in each group, then multiply by 100 to get the percentage.
# 5. Convert the churn rate series into a DataFrame for easier visualization by resetting the index and naming the new column 'churn_rate'.
# 6. Plot a bar chart to visualize the churn rate by region and gender:
#    - Use the 'region' column for the x-axis, 'churn_rate' for the y-axis, and 'gender' for the hue (legend) to differentiate between male and female.
#    - Set the title to 'Churn Rate by Region and Gender'.
#    - Label the x-axis as 'Region' and the y-axis as 'Churn Rate (%)'.


# Calculate the churn rate by region and gender

# Step 1: Group data by region and gender
grouped_data = df.groupby(['region', 'gender'])

# Step 2: Count total customers for each region-gender group
total_customers = grouped_data['exited'].count()

# Step 3: Count churned customers for each region-gender group
churned_customers = grouped_data['exited'].sum()  # 'exited' is 1 for churned customers

# Step 4: Calculate churn rate manually
churn_rate = (churned_customers / total_customers) * 100

# Step 5: Convert the series into a DataFrame for visualization
churn_rate_region_gender = churn_rate.reset_index(name='churn_rate')

# Step 6: Plot a bar chart to visualize churn rate by region and gender
plt.figure(figsize=(12, 6))
sns.barplot(x='region', y='churn_rate', hue='gender', data=churn_rate_region_gender)

# Step 7: Customize the plot
plt.title('Churn Rate by Region and Gender')
plt.xlabel('Region')
plt.ylabel('Churn Rate (%)')

# Step 8: Display the plot
plt.show()


### 9. Average satisfaction score for age groups

# 1. Filter the dataset to include only churned customers by selecting rows where 'exited' == 1.
# 2. Group the churned customers by 'age_group' and calculate the total number of churned customers in each group.
#    - Count the 'satisfaction_score' values for each 'age_group' to get the total churned customers per group.
# 3. Calculate the average satisfaction score for churned customers in each age group:
#    - Group the churned customers by 'age_group' and calculate the mean of the 'satisfaction_score' column.
#    - Reset the index and name the new column 'avg_satisfaction_score'.
# 4. Drop any rows where the 'avg_satisfaction_score' is NaN to clean up the data.
# 5. (Optional) Remove unused categories from the 'age_group' column if it's a categorical column with unused categories.
# 6. Print the resulting DataFrame to verify the data.
# 7. Plot a bar chart to visualize the average satisfaction score for churned customers by age group:
#    - Use 'age_group' for the x-axis and 'avg_satisfaction_score' for the y-axis.
#    - Set the title to 'Churned Customer Satisfaction by Age Group'.
#    - Label the x-axis as 'Age Group' and the y-axis as 'Average Satisfaction Score'.


# Step 1: Filter for churned customers where 'exited' == 1
churned_customers = df[df['exited'] == 1]

# Group by 'age_group' and calculate the average satisfaction score for churned customers
# Step 2: Count total churned customers per age group
total_churned_per_age_group = churned_customers.groupby('age_group')['satisfaction_score'].count()

# Step 3: Calculate the mean of satisfaction scores per age group
avg_satisfaction_churned = churned_customers.groupby('age_group')['satisfaction_score'].mean().reset_index(name="avg_satisfaction_score")

avg_satisfaction_churned = avg_satisfaction_churned.dropna(subset=['avg_satisfaction_score'])
# Remove unused categories from the 'age_group' column
# avg_satisfaction_churned['age_group'] = avg_satisfaction_churned['age_group'].cat.remove_unused_categories()
# print(avg_satisfaction_churned)
# Assuming df has 'age_group' and 'satisfaction_score' columns
plt.figure(figsize=(12, 6))
sns.barplot(x='age_group', y='avg_satisfaction_score', data=avg_satisfaction_churned)
plt.title('Churned Customer Satisfaction by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Satisfaction Score')
plt.show()



### 10. Count of customers by employment type, highlighting churned vs non-churned customers.

# 1. Group the data by 'employment_type' and 'exited' (churn status) and count the occurrences of each combination:
#    - Use groupby() to group by both 'employment_type' and 'exited', then apply size() to count the occurrences in each group.
#    - Reset the index of the result to make it easier to work with.
# 2. Create a count plot to visualize the relationship between 'employment_type' and churn status:
#    - Use 'employment_type' for the x-axis and 'exited' (churn status) for the hue to differentiate between churned and non-churned customers.
# 3. Customize the plot:
#    - Set the title to "Employment Type vs Churn".
#    - Label the x-axis as "Employment Type" and the y-axis as "Number of customers".


# Get numeric counts for each employment type and churn status
counts = df.groupby(['employment_type', 'exited']).size().reset_index()
# print(counts)
# Count Plot: Employment Type vs. Churn
plt.figure(figsize=(10, 6))
sns.countplot(x='employment_type', hue='exited', data=df, palette='coolwarm')
plt.title("Employment Type vs Churn")
plt.xlabel("Employment Type")
plt.ylabel("Number of customers")
plt.show()

### 11. Churn rates by region using Plotly

# 1. Group the data by 'state' using the groupby() function to categorize the customers by their region (state).
# 2. Count the total number of customers in each state by grouping by 'state' and counting the 'exited' column.
# 3. Count the number of churned customers in each state by grouping by 'state' and summing the 'exited' column (since 'exited' is 1 for churned customers).
# 4. Calculate the churn rate for each state manually:
#    - Divide the number of churned customers by the total number of customers in each state, then multiply by 100 to get the percentage.
# 5. Convert the churn rate series into a DataFrame for easier visualization by resetting the index and naming the new column 'churn_rate'.
# 6. Create a bar chart to visualize the churn rate by state using Plotly:
#    - Use the 'state' column for the x-axis and 'churn_rate' for the y-axis.
#    - Set the chart title to 'Churn Rate by State'.


import plotly.express as px
import plotly.io as pio
# Set default renderer for Plotly
pio.renderers.default = 'iframe'
# Step 1: Group data by state
grouped_data = df.groupby('state')

# Step 2: Count total customers per state
total_customers_by_state = grouped_data['exited'].count()

# Step 3: Count churned customers per state
churned_customers_by_state = grouped_data['exited'].sum()  # 'exited' is 1 for churned customers

# Step 4: Calculate churn rate manually
churn_rate_by_state = (churned_customers_by_state / total_customers_by_state) * 100

# print(churn_rate_by_state)

# Step 5: Convert to DataFrame for visualization
churn_rate_by_state_df = churn_rate_by_state.reset_index(name="churn_rate")

# print(churn_rate_by_state_df)

# Step 6: Create a bar chart to show churn rate by state
fig_bar = px.bar(churn_rate_by_state_df, x='state', y='churn_rate', title="Churn Rate by State")
fig_bar.show()


## Product Analysis

Why product analysis is important?

- Bank wants to understand why some of its customers are leaving. 
- They suspect that customers who use fewer products (like credit cards, loans, or savings accounts) are more likely to churn.
- They aim to identify which products are most linked to churn to create targeted retention strategies

Eg: A customer using both a credit card and a loan might be more likely to stay because they have multiple touchpoints with the bank.

df.columns

### 1.  Credit card distribution among churned customers

# 1. Filter the dataset to include only customers who have exited, i.e., where the 'exited' column is equal to 1. 
# 2. Create a count plot to visualize the distribution of credit card ownership among the exited customers.
#    - Use a seaborn countplot to show how many exited customers have a credit card (column 'hascrcard', where 0 = No, 1 = Yes).
#    - Apply a color palette ("coolwarm") for the plot to visually differentiate between categories.
# 3. Set the plot title to "Credit Card Ownership Distribution Among Exited Customers" to describe the plot's content.
# 4. Label the x-axis as "Has Credit Card (0 = No, 1 = Yes)" to clarify what the x-axis represents.
# 5. Label the y-axis as "Number of churned Customers" to explain what the y-axis represents.
# 6. Display the plot using plt.show() to visualize the results.



# Filter the dataset where 'isexited' == 1 (indicating customers who have exited)
exited_customers = df[df['exited'] == 1]

# # Create a count plot for credit card ownership among exited customers
plt.figure(figsize=(6, 4))
sns.countplot(x=exited_customers["hascrcard"],palette="coolwarm")
plt.title("Credit Card Ownership Distribution Among Exited Customers")
plt.xlabel("Has Credit Card (0 = No, 1 = Yes)")
plt.ylabel("Number of churned Customers")
plt.show()


# 1. Group the data by 'hascrcard' and 'exited' to count the number of customers in each group.
# 2. Calculate the total number of customers.
# 3. Calculate the percentage of each group based on the total number of customers.
# 4. Create a bar plot to visualize the churn percentage by credit card ownership.
# 5. Set the plot title to 'Global Customer Churn Percentage by Credit Card Status'.
# 6. Label the x-axis as 'Has Credit Card (0 = No, 1 = Yes)' and the y-axis as 'Percentage of Total Customers'.
# 7. Add a legend with the title 'Churned (Exited)'.
# 8. Display the plot.


# Step 1: Group by 'hascrcard' and 'exited' and count number of customers in each group
group_counts = df.groupby(['hascrcard', 'exited']).size().reset_index(name='count')

print(group_counts)

# Step 2: Calculate total number of customers
total_customers = group_counts['count'].sum()
#
print(total_customers)

# Step 3: Calculate percentage as (group count / total count) * 100
group_counts['percentage'] = (group_counts['count'] / total_customers) * 100

# Step 4: Plot using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(data=group_counts, x='hascrcard', y='percentage', hue='exited', palette="coolwarm", errorbar=None)

# Enhancing the plot
plt.title('Global Customer Churn Percentage by Credit Card Status')
plt.xlabel('Has Credit Card (0 = No, 1 = Yes)')
plt.ylabel('Percentage of Total Customers')
plt.legend(title='Churned (Exited)')
plt.tight_layout()
plt.show()

###  2. Credit card type distribution across churn status

# 1. Group the data by 'card_type' and 'exited' to count the number of customers in each group.
# 2. Calculate the total number of customers.
# 3. Calculate the percentage of each group and round the result to two decimal places.
# 4. Create a bar plot to visualize the percentage of customers by card type and churn status.
# 5. Set the plot title to 'Credit card type percentage across churn status'.
# 6. Label the x-axis as 'Credit card type' and the y-axis as 'Percentage of Total Customers'.
# 7. Display the plot.


# Step 1: Group by 'card_type' and 'exited' and count number of customers in each group
card_counts = df.groupby(['card_type', 'exited']).size().reset_index(name='count')

# Step 2: Calculate total number of customers
total_customers = card_counts['count'].sum()

# Step 3: Calculate percentage as (group count / total count) * 100 and round
card_counts['percentage'] = ((card_counts['count'] / total_customers) * 100).round(2)

print(card_counts)

# Step 4: Plot using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(data=card_counts, x='card_type', y='percentage', hue='exited', palette="coolwarm", errorbar=None)

# Enhancing the plot
plt.title('Credit card type percentage across churn status')
plt.xlabel('Credit card type')
plt.ylabel('Percentage of Total Customers')
plt.tight_layout()
plt.show()


###  3.  Loan ownership vs Churn

# 1. Create a count plot to visualize loan ownership distribution by churn status.
# 2. Use 'hasloan' for the x-axis and 'exited' for the hue (churn status).
# 3. Set the plot title to 'HasLoan Ownership Distribution by Churned Status'.
# 4. Label the x-axis as 'HasLoan (0 = No, 1 = Yes)' and the y-axis as 'Number of Customers'.
# 5. Display the plot.


plt.figure(figsize=(6, 4))
sns.countplot(x=df["hasloan"], hue=df["exited"], palette="coolwarm")
plt.title("HasLoan Ownership Distribution by Churned Status")
plt.xlabel("HasLoan (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()


###  4. Number of products vs churn

# 1. Filter the dataset to include only customers who have exited (where 'exited' == 1).
# 2. Create a count plot to visualize the distribution of credit card types among exited customers.
# 3. Set the plot title to 'Credit Card Type Distribution'.
# 4. Label the x-axis as 'Card type' and the y-axis as 'No of churned customers'.
# 5. Display the plot.



# Filter the dataset where 'isexited' == 1 (indicating customers who have exited)
exited_customers = df[df['exited'] == 1]

# Credit Card Type Distribution
# plt.figure(figsize=(8, 4))
sns.countplot(x=exited_customers["card_type"], palette="coolwarm")
plt.title("Credit Card Type Distribution")
plt.xlabel("Card type")
plt.ylabel("No of churned customers")
plt.show()


###  5.  Product distribution across churn 

# 1. Group the dataset by churn status ('exited') and each product ('hascrcard', 'hasloan', 'hasfd') to count the number of customers for each.
# 2. Combine the counts into a DataFrame for easier plotting.
# 3. Create a bar plot to visualize the product distribution across churn status.
# 4. Annotate each bar in the plot with the count of customers.
# 5. Set the plot title to 'Product Distribution Across Churn Status', label the axes, and display the legend.
# 6. Display the plot.


# Step 1: Group by churn status and each product
crcard = df[df['hascrcard'] == 1].groupby('exited').size()
print("Credit card: ")
print(crcard)

loan = df[df['hasloan'] == 1].groupby('exited').size()
print("Loan count: ")
print(loan)

fd = df[df['hasfd'] == 1].groupby('exited').size()
print("HasLoan count: ")
print(fd)

# Step 2: Combine into a DataFrame
product_dist = pd.DataFrame({
    'Credit Card': crcard,
    'Loan': loan,
    'Fixed Deposit': fd
}) 

# Step 1: Plot the bar chart
ax = product_dist.plot(kind='bar', figsize=(5, 6))

# Step 2: Automatically label bars
ax.bar_label(ax.containers[0], fontsize=9, padding=3)
ax.bar_label(ax.containers[1], fontsize=9, padding=3)
ax.bar_label(ax.containers[2], fontsize=9, padding=3)


plt.title('Product Distribution Across Churn Status')
plt.xlabel('Churn status')
plt.ylabel('Number of Customers')
plt.legend(title='Products')
plt.tight_layout()
plt.show()


###  6. Average product usage by tenure groups across churn status

# 1. Create tenure groups by categorizing the 'tenure' column into bins of 5 years.
# 2. Create a bar plot to visualize the average number of products across tenure groups and churn status.
# 3. Set the x-axis label to 'Tenure Group (Years)' and the y-axis label to 'Average Number of Products'.
# 4. Set the plot title to 'Tenure Group vs Avg Number of Products by Churn Status'.
# 5. Add a legend indicating 'Exited' status, with labels 'No' and 'Yes'.
# 6. Display the plot.



# Step 1: Create tenure groups ---> Make this step simple by using array of value instead
tenure_bins = list(range(0, 41, 5))
tenure_labels = [f'{i}<={i+5}' for i in range(0, 36, 5)]
df['tenure_group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, right=False)

# Step 2: Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df,
    x='tenure_group',
    y='numofproducts',
    hue='exited',
    palette='muted',
    errorbar=None
)
plt.xlabel('Tenure Group (Years)')
plt.ylabel('Average Number of Products')
plt.title('Tenure Group vs Avg Number of Products by Churn Status')
plt.legend(title='Exited', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


###  7.  Average number of products usage by given credit score ranges across churned status

# 1. Create credit score bins to categorize the 'creditscore' column into specified ranges.
# 2. Create a bar plot to visualize the average number of products across credit score ranges and churn status.
# 3. Annotate each bar with the average value of the 'numofproducts'.
# 4. Set the plot title to 'Average Product Usage by Credit Score Range and Churn Status'.
# 5. Label the x-axis as 'Credit Score Range' and the y-axis as 'Average Number of Products'.
# 6. Add a legend indicating churn status, with labels 'Churned (1 = Yes, 0 = No)'.
# 7. Display the plot.


# Step 1: Create credit score bins

credit_bins = [200, 400, 600, 800, 1000, 1200]
credit_labels = ['200-399', '400-599', '600-799', '800-999', '1000-1199']

df['credit_score_range'] = pd.cut(df['creditscore'], bins=credit_bins, labels=credit_labels)

# Step 2: Set figure size
# plt.figure(figsize=(10, 6))

# Step 3: Create a bar plot for credit score range vs. number of products across churned status
ax = sns.barplot(
    data=df,
    x='credit_score_range',
    y='numofproducts',
    hue='exited',
    palette='coolwarm',
    errorbar=None
)

# Step 4: Annotate each bar with average value
for p in ax.patches:
    height = p.get_height()
    if pd.notnull(height) and height > 0:
        ax.text(
            x=p.get_x() + p.get_width() / 2,
            y=height + 0.02,
            s=f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )


# Step 4: Label the axes and title
plt.title('Average Product Usage by Credit Score Range and Churn Status')
plt.xlabel('Credit Score Range')
plt.ylabel('Average Number of Products')
plt.legend(title='Churned (1 = Yes, 0 = No)')
plt.tight_layout()
plt.show()


df.columns

## Feedback Analysis

### 🧠 Assumption 1:
Customers with low satisfaction and low loyalty points are more likely to churn.

✅ Approach:
- You're plotting two continuous variables: satisfaction_score and point_earned.
- A scatter plot helps detect relationships, clusters, or patterns between these two variables.
- Coloring by churn_status adds a third visual dimension — allowing us to visually detect trends in churn behavior based on combinations of satisfaction and rewards.


# Map 'exited' column to categorical labels
df["churn_status"] = df["exited"].map({0: "Stayed", 1: "Left"})
df['churn_status']



### 🔍 **How `.map()` Works (with Example)**

Let’s break it down with your line:

```python
df["churn_status"] = df["exited"].map({0: "Stayed", 1: "Left"})
```

#### 🧠 What it does:
- `df["exited"]` is a column with numeric values: `0` (not churned), `1` (churned)
- `.map({0: "Stayed", 1: "Left"})` **replaces**:
  - `0` → `"Stayed"`
  - `1` → `"Left"`
- The result is a new column `churn_status` with more readable labels



# TODO 1: 
# - Use satisfaction_score as the X-axis.
# - Use point_earned as the Y-axis.
# - Color-code the points using the churn_status column (Stayed vs Left).
# - Use a scatter plot to show all points.
# - Chart title: Satisfaction vs Points Earned by Churn
color_map  = {"Stayed": "blue", "Left": "red"}
colors = df["churn_status"].map(color_map)
plt.scatter(df['satisfaction_score'] ,df['point_earned'] ,c =  colors )
for status, color in color_map.items():
    plt.scatter([], [], c=color, label=status)

plt.title('Satisfaction vs Points Earned by Churn')
plt.legend()
plt.show()

### 🧠 Assumption 2:
Customers who churn tend to have lower satisfaction scores than those who stay.


✅ Approach:
- Compare the distribution of satisfaction_score for churned vs non-churned customers.
- Plot both on the same axis for visual comparison.

📊 Why this chart?
- A histogram shows how a variable is spread out.
- When overlaid, we can spot shifts in the score distribution between groups.

# TODO 2: Plot the histogram of satisfaction scores
# - Set 'satisfaction_score' as the variable on the x-axis
# - Use 'churn_status' as the hue to separate churned vs retained customers
# - Set number of bins to 10
# - Use 'stack' to stack churn categories on top of each other
# - Chart title: Satisfaction Score Distribution by Churn


sns.histplot(x ='satisfaction_score' ,hue = "churn_status" ,multiple="stack", bins = 10, data = df)
plt.title('Satisfaction Score Distribution by Churn')
plt.show()

### 🧠 Assumption 3:
Among those who complained, a higher proportion ended up churning.

✅ Approach:
- Focus only on rows where complain = 1
- Show what portion of them stayed vs left

📊 Why this chart?
- A pie chart visually highlights proportions within a group — perfect for categorical splits like churn within complainants.

# TODO

# - Select only the row where complain = 1 (complainers).
# - Use .value_counts() and Normalize the values to get proportions (not raw counts).
# - Plot a pie chart for churn distribution among those who complained.
#  chart title: Churn Distribution Among Complainers

new_df = df[df['complain']==1]
new_df['exited'] = df['exited'].map({0:'Stayed' , 1:'Left'})
complain_ana = new_df.groupby('exited')['complain'].value_counts().reset_index()
print(complain_ana)

plt.pie(
    complain_ana['count'],
    labels=complain_ana['exited'],
    autopct='%1.1f%%',
    colors=['blue', 'red']
)

plt.title('Churn Distribution Among Complainers')
plt.show()

# Step 1: Calculate IQR
import numpy as np
Q1 = df["balance"].quantile(0.25)
Q3 = df["balance"].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define lower and upper caps
lower_cap = Q1 - 1.5 * IQR
upper_cap = Q3 + 1.5 * IQR

# Step 3: Modify the original column directly
df["balance"] = np.where(df["balance"] > upper_cap, upper_cap, df["balance"])

# Step 1: Calculate IQR
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define lower and upper caps
lower_cap = Q1 - 1.5 * IQR
upper_cap = Q3 + 1.5 * IQR

# Step 3: Modify the original column directly
df["salary"] = np.where(df["salary"] > upper_cap, upper_cap, df["salary"])


### Financial Analysis

### Assumption 1
Customers with high salary and high balance are less likely to churn
Approach:
- You’re dealing with two continuous numerical variables: salary and balance.
- A scatter plot is ideal to visualize how those two interact and to detect patterns or clusters related to churn.
- Coloring by churn_status adds an extra dimension (a third variable) visually, making it easier to spot correlations between financial metrics and churn behavior.


# TODO
# Extract sample data points for visualization using Random Sampling
# Create a scatter plot:
# - Plot salary on the x-axis and balance on the y-axis.
# - Use churn_status as the hue to differentiate churned vs. retained customers.
# - Add transparency (alpha=0.6) to help visualize overlapping points.
# Chart title: Salary vs Balance by Churn Status
df_sample = df.sample(n=500, random_state=42)
df_sample
# df['churn_typeeee'] = df['exited'].map({0:'Stayed' , 1:'Left'})
# print(df[['exited' , 'churn_typeeee']])
sns.scatterplot(data =df_sample , x = 'salary' ,y = 'balance' ,hue = 'churn_status' ,alpha =0.6)
plt.title('Salary vs Balance by Churn Status')
plt.show()

### Assumption 2
- Customers with more products and lower balances are more likely to churn

Approach:
- Group the data by number of products (numofproducts) and churn status (churn_status).
- Calculate the average balance for each group.
- Use a bar plot to visualize the average balance per product count, with churn status shown as different bars (hue).
- This allows for side-by-side comparisons across product levels to spot trends.


# TODO
# Group data by number of products and churn status:
# Create a bar plot:
# - Plot numofproducts on the x-axis and the average balance on the y-axis.
# - Use churn_status as the hue to compare churned vs. retained customers side by side within each product count group.
# - Chart Title: Average Balance by Number of Products and Churn Status

group = df.groupby(['numofproducts' , 'churn_status'])['balance'].mean().reset_index()
group


sns.barplot(x = 'numofproducts' ,y ='balance',hue = 'churn_status' , data =group  )
plt.title('Average Balance by Number of Products and Churn Status')
plt.show()

### Assumption 3
- Churn rate differs between FD holders and non-holders

Approach:
- Filter the dataset to include only FD holders (hasfd == 1).
- Use value_counts(normalize=True) gives percentage proportions of churned vs. non-churned customers among FD holders.
- Plot pie chart to show part-to-whole relationships.


# TODO
# Filter the dataset to include only FD holders
# Use value_counts(normalize=True) on churn_status to get the proportion of churned vs. non-churned customers.
# Round the values for cleaner display.
# Plot the pie chart:
# - Use the churn distribution to create a pie chart.
# - Chart title : Churn Distribution Among FD Holders

# df.columns

new_df = df[df['hasfd'] ==1]
churn_proportion = df['churn_status'].value_counts(normalize=True)
# print(new_df)
plt.pie(churn_proportion , labels = churn_proportion.index , autopct='%1.1f%%' )
plt.title('Churn Distribution Among FD Holders')
plt.show()

