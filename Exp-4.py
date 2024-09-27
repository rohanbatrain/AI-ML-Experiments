# Write a python program to import and export data using Pandas and 
# show the details of the dataset like number of rows, columns, first five 
# rows, size, number of missing values, sum, average, min and max 
# values from the numerical columns.

import pandas as pd

# Import data from a CSV file
df = pd.read_csv('student-scores.csv')

# Display dataset details
# Number of Rows
print("Number of rows:", df.shape[0])
# Number of columns
print("Number of columns:", df.shape[1])

# printing 5 rows 
print("\nFirst five rows:\n", df.head())
# Size of Dataset
print("\nSize of the dataset:", df.size)
# Cheking if there is any missing values in Data set
print("\nNumber of missing values:\n", df.isnull().sum())

# Display summary statistics for numerical columns
# summision of all numerical values present in Dataset
print("\nSum of numerical columns:\n", df.sum(numeric_only=True))
# Average of all numerical values present in Dataset
print("\nAverage of numerical columns:\n", df.mean(numeric_only=True))
# minimal values in all numerical values present in Dataset
print("\nMinimum values of numerical columns:\n", df.min(numeric_only=True))
# minimal values in all numerical values present in Dataset
print("\nMaximum values of numerical columns:\n", df.max(numeric_only=True))

# Export data to a new CSV file
df.to_csv('output.csv', index=False)
