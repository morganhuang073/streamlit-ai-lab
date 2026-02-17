"""
Streamlit app for exploratory data analysis of the Iris dataset.
This app allows users to visualize and analyze the Iris dataset interactively.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(page_title="Iris EDA", layout="wide")

# Title and description
st.title("📊 Iris Dataset Exploratory Data Analysis")
st.markdown("Explore and visualize the classic Iris dataset with interactive tools.")

# Load the Iris dataset
@st.cache_data
def load_data():
    """Load the Iris dataset and return as a pandas DataFrame."""
    iris_data = load_iris()
    iris_df = pd.DataFrame(
        data=iris_data.data,
        columns=iris_data.feature_names
    )
    # Add the target column (species)
    iris_df['Species'] = iris_data.target_names[iris_data.target]
    return iris_df

# Load data
df = load_data()

# Display dataset overview section
st.header("Dataset Overview")

# Display first rows
st.subheader("First Few Rows")
num_rows = st.slider("Number of rows to display:", min_value=1, max_value=20, value=5)
st.dataframe(df.head(num_rows), use_container_width=True)

# Display dataset info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", len(df))
with col2:
    st.metric("Total Columns", len(df.columns))
with col3:
    st.metric("Unique Species", df['Species'].nunique())

# Display summary statistics section
st.header("Summary Statistics")
st.subheader("Descriptive Statistics")
st.dataframe(df.describe(), use_container_width=True)

# Interactive column selection section
st.header("Interactive Visualizations")

# Get numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Column selection for histogram
st.subheader("Histogram Analysis")
selected_histogram_column = st.selectbox(
    "Select a column for histogram:",
    options=numeric_columns,
    key="histogram_column"
)

# Display histogram
fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
ax_hist.hist(df[selected_histogram_column], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax_hist.set_xlabel(selected_histogram_column)
ax_hist.set_ylabel("Frequency")
ax_hist.set_title(f"Distribution of {selected_histogram_column}")
ax_hist.grid(axis='y', alpha=0.3)
st.pyplot(fig_hist)

# Column selection for scatter plot
st.subheader("Scatter Plot Analysis")
col1_scatter, col2_scatter = st.columns(2)

with col1_scatter:
    selected_x_column = st.selectbox(
        "Select X-axis column:",
        options=numeric_columns,
        key="scatter_x"
    )

with col2_scatter:
    selected_y_column = st.selectbox(
        "Select Y-axis column:",
        options=numeric_columns,
        index=1 if len(numeric_columns) > 1 else 0,
        key="scatter_y"
    )

# Display scatter plot with species color coding
fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

# Create scatter plot with different colors for each species
species_list = df['Species'].unique()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for species, color in zip(species_list, colors):
    species_data = df[df['Species'] == species]
    ax_scatter.scatter(
        species_data[selected_x_column],
        species_data[selected_y_column],
        label=species,
        alpha=0.7,
        s=100,
        color=color,
        edgecolors='black',
        linewidth=0.5
    )

ax_scatter.set_xlabel(selected_x_column)
ax_scatter.set_ylabel(selected_y_column)
ax_scatter.set_title(f"{selected_x_column} vs {selected_y_column}")
ax_scatter.legend()
ax_scatter.grid(alpha=0.3)
st.pyplot(fig_scatter)

# Display raw data section
st.header("Data Inspector")
if st.checkbox("Show full dataset"):
    st.dataframe(df, use_container_width=True)

# Filter by species section
st.subheader("Filter by Species")
selected_species = st.multiselect(
    "Select species to display:",
    options=df['Species'].unique(),
    default=df['Species'].unique()
)

filtered_df = df[df['Species'].isin(selected_species)]
st.dataframe(filtered_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Created with Streamlit | Data source: scikit-learn Iris dataset")
