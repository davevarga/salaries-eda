import math
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'bisque', 'aquamarine','plum', 'lightpink', 'sandybrown']

# Custom color map: gradient for green (closer to 0) and red (closer to 1)
colors_gradient = {
    "red": [(0.0, "white"), (0.5, "lightcoral"), (1.0, "darkred")],
    "green": [(0.0, "white"), (0.5, "lightgreen"), (1.0, "darkgreen")],
}

# Combine the two gradients into one custom colormap
red_green_cmap = LinearSegmentedColormap.from_list(
    "red_green",
    [(0, "darkgreen"), (0.5, "white"), (1, "darkred")],
    N=256
)

def plot_distribution(dataset, column, bins=20):
    """
    Plots the distribution curve of a given column and displays the standard deviation
    and variance on the plot.

    Args:
        dataset (pandas.DataFrame): The dataset containing the column to plot.
        column (str): The column name to plot the distribution for.
    """
    # Calculate the variance and standard deviation
    variance = np.var(dataset[column], ddof=1)  # Sample variance
    std_dev = np.std(dataset[column], ddof=1)   # Sample standard deviation

    # Remove outlier values after calculating variance and std. deviation
    dataset = remove_outliers(dataset, column)

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.histplot(dataset[column], kde=True, bins=bins, color='blue', stat='density')

    # Add the standard deviation and variance text on the plot
    plt.title(f'Distribution of {column}', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.text(0.95, 0.85, f'Variance: {variance:.2f}\nStd Dev: {std_dev:.2f}',
             horizontalalignment='right', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12, color='black')

    plt.show()


def plot_distributions_grid(dataset, columns, layout=(2, 2), bins=20, figsize=(15, 10)):
    """
    Plots distribution curves for multiple columns in a grid layout.
    """
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the grid for easier iteration

    for i, column in enumerate(columns):
        # Remove outliers after calculating variance and std. deviation
        dataset = remove_outliers(dataset, column)

        data = dataset[column].dropna()

        std_dev = np.std(data)
        variance = np.var(data)

        sns.histplot(data, kde=True, bins=bins, color=colors[i], stat="density", ax=axes[i], linewidth=0)
        axes[i].set_title(f'Distribution of {column}', fontsize=12)
        axes[i].set_xlabel(column, fontsize=10)
        axes[i].set_ylabel('Density', fontsize=10)

        # Annotate standard deviation and variance
        axes[i].text(0.95, 0.95, f'Std Dev: {std_dev:.2f}\nVariance: {variance:.2f}',
                     horizontalalignment='right', verticalalignment='top',
                     transform=axes[i].transAxes, fontsize=8, color='black')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_group_counts(dataset, columns, grid_shape=(2, 2), figsize=(16, 9)):
    """
    Plots bar charts for group counts in the specified categorical columns.

    Args:
        dataset (pandas.DataFrame): The dataset containing the data.
        columns (list of str): List of categorical column names to plot.
        grid_shape (tuple, optional): Tuple (rows, cols) defining the grid layout.
                                      If not provided, it is automatically determined.
    Returns:
        None
    """
    # Determine the number of plots
    num_plots = len(columns)

    # Determine grid layout if not specified
    if not grid_shape:
        cols = min(2, num_plots)  # Default to at most 2 columns
        rows = math.ceil(num_plots / cols)
        grid_shape = (rows, cols)

    # Create the figure and axes
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    axes = axes.flatten()  # Flatten to handle 1D indexing

    for idx, col in enumerate(columns):
        # Plot the bar chart for the current column
        sns.countplot(data=dataset, x=col, ax=axes[idx], palette="muted", hue=col)
        axes[idx].set_title(f"Count by {col}", fontsize=14)
        axes[idx].set_xlabel(col, fontsize=12)
        axes[idx].set_ylabel("Count", fontsize=12)

    # Hide unused subplots if there are any
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def remove_outliers(dataset, column):
    """
    Removes outliers from the specified column in the dataset using the IQR method.

    Args:
        dataset (pandas.DataFrame): The dataset to clean.
        column (str): The column name from which to remove outliers.

    Returns:
        pandas.DataFrame: The cleaned dataset with outliers removed.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)

    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Define the outlier thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers by filtering the dataset
    cleaned_dataset = dataset[(dataset[column] >= lower_bound) & (dataset[column] <= upper_bound)]

    return cleaned_dataset


def remove_duplicates(dataset, subset):
    # Remove duplicates
    cleaned_dataset = dataset.drop_duplicates(subset=subset)
    return cleaned_dataset


def standard_clean(dataframe, string_columns, map=None):
    # If a map between names is provided then rename columns
    if map is not None:
        dataframe.rename(columns=map, inplace=True)

    # Separate numeric and sting type variables
    numeric_columns = dataframe.drop(string_columns, axis=1).columns

    # Convert relevant columns to numeric, forcing errors to NaN where necessary
    for column in numeric_columns:
        dataframe[column] = dataframe[column].astype(str).str.replace(',', '.')
        dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')

    # Remove white spaces and standardize string format
    for column in string_columns:
        dataframe[column] = dataframe[column].str.strip()
        dataframe[column] = dataframe[column].str.lower()

    # Drop NaN values
    return dataframe


def calculate_iou(Y_pred, Y_norm):
    # Compute KDE (Kernel Density Estimate) for Actual and Predicted values
    kde_actual = gaussian_kde(Y_norm)  # KDE for actual values
    kde_predicted = gaussian_kde(Y_pred)  # KDE for predicted values

    # Define a common range of x values for integration
    x_values = np.linspace(min(Y_norm.min(), Y_pred.min()), max(Y_norm.max(), Y_pred.max()), 1000)

    # Evaluate the densities at each x value
    actual_density = kde_actual(x_values)
    predicted_density = kde_predicted(x_values)

    # Compute the Intersection (min of the two densities at each point)
    intersection_density = np.minimum(actual_density, predicted_density)
    area_intersection = np.trapz(intersection_density, x_values)

    # Compute the Union (sum of the two densities minus their intersection)
    union_density = np.maximum(actual_density, predicted_density)
    area_union = np.trapz(union_density, x_values)

    # Intersection over Union (IoU)
    iou = area_intersection / area_union
    return iou