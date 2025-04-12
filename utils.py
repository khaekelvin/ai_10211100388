import streamlit as st
import pandas as pd
import numpy as np
import io

def display_dataframe_info(df):
    """
    Display key information about a dataframe
    
    Args:
        df: Pandas DataFrame to analyze
    """
    # Display basic info
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Column 1: Data types
    with col1:
        dtype_counts = df.dtypes.value_counts().reset_index()
        dtype_counts.columns = ["Data Type", "Count"]
        st.write("**Data Types:**")
        st.dataframe(dtype_counts)
    
    # Column 2: Missing values
    with col2:
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_data = missing_data.sort_values('Missing Values', ascending=False)
        
        st.write("**Missing Values (Top 10):**")
        st.dataframe(missing_data.head(10))
    
    # Numerical columns statistics
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        st.write("**Numerical Columns Statistics:**")
        st.dataframe(df[num_cols].describe().round(2))
    
    # Categorical columns statistics
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    
    if len(cat_cols) > 0:
        st.write("**Categorical Columns:**")
        
        # Show the number of unique values for each categorical column
        cat_stats = pd.DataFrame({
            'Column': cat_cols,
            'Unique Values': [df[col].nunique() for col in cat_cols],
            'Mode': [df[col].mode()[0] if not df[col].mode().empty else 'N/A' for col in cat_cols],
            'Mode Frequency': [df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0 for col in cat_cols],
            'Mode %': [(df[col].value_counts().iloc[0] / len(df) * 100).round(2) if not df[col].value_counts().empty else 0 for col in cat_cols]
        })
        
        st.dataframe(cat_stats)

def scale_dataframe(df, columns, method='standard'):
    """
    Scale selected columns in a dataframe
    
    Args:
        df: Pandas DataFrame
        columns: List of column names to scale
        method: Scaling method ('standard', 'minmax', or 'robust')
        
    Returns:
        Scaled DataFrame
    """
    # Create a copy to avoid modifying the original
    scaled_df = df.copy()
    
    if method == 'standard':
        # Standardize to mean=0, std=1
        for col in columns:
            mean = scaled_df[col].mean()
            std = scaled_df[col].std()
            scaled_df[col] = (scaled_df[col] - mean) / std
    
    elif method == 'minmax':
        # Scale to 0-1 range
        for col in columns:
            min_val = scaled_df[col].min()
            max_val = scaled_df[col].max()
            scaled_df[col] = (scaled_df[col] - min_val) / (max_val - min_val)
    
    elif method == 'robust':
        # Scale based on median and interquartile range
        for col in columns:
            median = scaled_df[col].median()
            q1 = scaled_df[col].quantile(0.25)
            q3 = scaled_df[col].quantile(0.75)
            iqr = q3 - q1
            scaled_df[col] = (scaled_df[col] - median) / iqr
    
    return scaled_df

def encode_categorical_features(df, columns, method='onehot'):
    """
    Encode categorical features in a dataframe
    
    Args:
        df: Pandas DataFrame
        columns: List of column names to encode
        method: Encoding method ('onehot', 'label', or 'ordinal')
        
    Returns:
        Encoded DataFrame
    """
    # Create a copy to avoid modifying the original
    encoded_df = df.copy()
    
    if method == 'onehot':
        # One-hot encoding
        encoded_df = pd.get_dummies(encoded_df, columns=columns, drop_first=False)
    
    elif method == 'label':
        # Label encoding
        for col in columns:
            encoded_df[col] = encoded_df[col].astype('category').cat.codes
    
    elif method == 'ordinal':
        # Ordinal encoding (requires additional mapping)
        # This is a simplified implementation
        for col in columns:
            unique_values = sorted(encoded_df[col].unique())
            mapping = {val: i for i, val in enumerate(unique_values)}
            encoded_df[col] = encoded_df[col].map(mapping)
    
    return encoded_df

def handle_missing_values(df, strategy='mean', columns=None, fill_value=None):
    """
    Handle missing values in a dataframe
    
    Args:
        df: Pandas DataFrame
        strategy: Strategy for handling missing values 
                ('mean', 'median', 'mode', 'drop', or 'fill')
        columns: List of column names to process (None for all columns)
        fill_value: Value to use for 'fill' strategy
        
    Returns:
        DataFrame with missing values handled
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, use all columns
    if columns is None:
        columns = df.columns
    
    if strategy == 'drop':
        # Drop rows with missing values in specified columns
        result_df = result_df.dropna(subset=columns)
    
    else:
        # Apply strategy to each column
        for col in columns:
            if col in result_df.columns:
                if strategy == 'mean' and result_df[col].dtype in ['int64', 'float64']:
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                
                elif strategy == 'median' and result_df[col].dtype in ['int64', 'float64']:
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                
                elif strategy == 'mode':
                    mode_value = result_df[col].mode()
                    result_df[col] = result_df[col].fillna(mode_value[0] if not mode_value.empty else None)
                
                elif strategy == 'fill' and fill_value is not None:
                    result_df[col] = result_df[col].fillna(fill_value)
    
    return result_df

def create_download_link(df, filename="data.csv", link_text="Download data as CSV"):
    """
    Create a download link for a dataframe
    
    Args:
        df: Pandas DataFrame to download
        filename: Name of the downloaded file
        link_text: Text to display for the download link
        
    Returns:
        HTML link for downloading the data
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=link_text,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def get_model_metrics_df(metrics_dict, format_percentage=True):
    """
    Create a dataframe for model metrics
    
    Args:
        metrics_dict: Dictionary of metrics
        format_percentage: Whether to format percentage values
        
    Returns:
        DataFrame with metrics
    """
    metrics_df = pd.DataFrame({
        'Metric': list(metrics_dict.keys()),
        'Value': list(metrics_dict.values())
    })
    
    if format_percentage:
        # Format metrics that are typically percentages
        percentage_metrics = ['accuracy', 'precision', 'recall', 'f1', 'r2']
        
        for metric in percentage_metrics:
            if metric in metrics_dict:
                idx = metrics_df[metrics_df['Metric'] == metric].index
                if len(idx) > 0:
                    metrics_df.loc[idx, 'Value'] = metrics_df.loc[idx, 'Value'].apply(lambda x: f"{x:.2%}")
    
    return metrics_df

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Plot a confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: Class labels
        title: Plot title
    """
    import plotly.figure_factory as ff
    
    # Create annotated heatmap
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=classes,
        y=classes,
        annotation_text=cm,
        colorscale='Blues'
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Predicted'),
        yaxis=dict(title='Actual'),
        width=500,
        height=500
    )
    
    return fig

def convert_df_to_csv(df):
    """
    Convert a dataframe to CSV for download
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        CSV string
    """
    return df.to_csv(index=False).encode('utf-8')

def highlight_important_features(feature_importances, top_n=10):
    """
    Format feature importance data for visualization
    
    Args:
        feature_importances: Dictionary with feature names as keys and importance values
        top_n: Number of top features to highlight
        
    Returns:
        DataFrame with formatted feature importances
    """
    # Convert to dataframe
    fi_df = pd.DataFrame({
        'Feature': list(feature_importances.keys()),
        'Importance': list(feature_importances.values())
    })
    
    # Sort by importance
    fi_df = fi_df.sort_values('Importance', ascending=False)
    
    # Get top N features
    top_features = fi_df.head(top_n)
    
    return top_features
