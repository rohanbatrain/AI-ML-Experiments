# Experiment 08: Implementing PCA for Dimensionality Reduction
# Lab Experiment 08 B.Tech - Elements of AIML Lab
# Objective: Implement PCA for dimensionality reduction on a sample dataset

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load the Dataset
def load_dataset():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print("Dataset Loaded Successfully:\n", df.head())
    return df, data.target

# Step 2: Standardize the Data
def standardize_data(df):
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)
    print("Data Standardized:\n", df_standardized[:5])
    return df_standardized

# Step 3: Calculate Covariance Matrix
def calculate_covariance_matrix(df_standardized):
    cov_matrix = np.cov(df_standardized, rowvar=False)
    print("Covariance Matrix:\n", cov_matrix)
    return cov_matrix

# Step 4: Find Eigenvalues and Eigenvectors
def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
    return eigenvalues, eigenvectors

# Step 5: Choose the Number of Components
def choose_components(eigenvalues):
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
    print(f"Number of components chosen: {num_components}")
    return num_components

# Step 6: Transform the Data
def transform_data(df_standardized, eigenvectors, num_components):
    transformed_data = np.dot(df_standardized, eigenvectors[:, :num_components])
    print("Transformed Data (First 5 Rows):\n", transformed_data[:5])
    return transformed_data

# Step 7: Visualize the Results
def plot_transformed_data(transformed_data, target):
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=target, cmap='viridis')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Result (First Two Components)")
    plt.colorbar()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and standardize the dataset
    df, target = load_dataset()
    df_standardized = standardize_data(df)

    # Calculate covariance matrix and eigen decomposition
    cov_matrix = calculate_covariance_matrix(df_standardized)
    eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)

    # Choose the number of components and transform the data
    num_components = choose_components(eigenvalues)
    transformed_data = transform_data(df_standardized, eigenvectors, num_components)

    # Visualize the transformed data
    plot_transformed_data(transformed_data, target)

