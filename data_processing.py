import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils import display_dataframe_info

def render_data_upload_and_exploration(datasets):
    st.title("Data Upload & Exploration")
    
    st.markdown("""
    Upload your own dataset or use the pre-loaded Ghana election data for analysis.
    This section allows you to explore and preprocess your data before applying 
    machine learning techniques.
    """)
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ["Use Preloaded Ghana Election Data", "Upload Custom Dataset"]
    )
    
    if data_source == "Upload Custom Dataset":
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read accordingly
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                datasets['custom_data'] = df
                datasets['current_df'] = df
                
                st.success(f"Successfully loaded data from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return
        elif datasets['custom_data'] is not None:
            df = datasets['custom_data']
            datasets['current_df'] = df
        else:
            st.info("Please upload a file to continue.")
            return
    else:
        # Use preloaded election data
        if datasets['election_data'] is not None:
            df = datasets['election_data']
            datasets['current_df'] = df
        else:
            st.error("Preloaded election data is not available.")
            return
    
    # Display basic dataset information
    st.subheader("Dataset Overview")
    display_dataframe_info(df)
    
    # Data exploration tabs
    exploration_tab, preprocessing_tab = st.tabs(["Data Exploration", "Data Preprocessing"])
    
    with exploration_tab:
        st.subheader("Explore Your Data")
        
        # Show a sample of the data
        st.write("Data Sample:")
        st.dataframe(df.head(10))
        
        # Column selection for visualization
        if len(df.columns) > 0:
            # Determine numerical and categorical columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            # Visualization options
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"]
            )
            
            if viz_type == "Histogram" and len(numerical_cols) > 0:
                col = st.selectbox("Select a numerical column:", numerical_cols)
                fig = px.histogram(df, x=col, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Bar Chart":
                if len(categorical_cols) > 0 and len(numerical_cols) > 0:
                    x_col = st.selectbox("Select categorical column (x-axis):", categorical_cols)
                    y_col = st.selectbox("Select numerical column (y-axis):", numerical_cols)
                    
                    # Check if we need to aggregate the data
                    if df[x_col].nunique() > 15:
                        st.warning(f"Column {x_col} has {df[x_col].nunique()} unique values. Consider selecting a different column or using preprocessing to reduce categories.")
                    
                    agg_func = st.selectbox("Select aggregation function:", ["sum", "mean", "count", "min", "max"])
                    
                    # Create aggregated dataframe
                    agg_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index().sort_values(y_col, ascending=False).head(15)
                    
                    fig = px.bar(
                        agg_df, 
                        x=x_col, 
                        y=y_col, 
                        title=f"{agg_func.capitalize()} of {y_col} by {x_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient column types for bar chart (need both categorical and numerical columns)")
                
            elif viz_type == "Scatter Plot" and len(numerical_cols) >= 2:
                x_col = st.selectbox("Select x-axis (numerical):", numerical_cols)
                y_col = st.selectbox("Select y-axis (numerical):", [col for col in numerical_cols if col != x_col])
                
                color_col = None
                if len(categorical_cols) > 0:
                    use_color = st.checkbox("Add color dimension")
                    if use_color:
                        color_col = st.selectbox("Select color column (categorical):", categorical_cols)
                
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    color=color_col,
                    title=f"Scatter Plot: {y_col} vs {x_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Box Plot" and len(numerical_cols) > 0:
                y_col = st.selectbox("Select numerical column:", numerical_cols)
                
                x_col = None
                if len(categorical_cols) > 0:
                    use_category = st.checkbox("Group by category")
                    if use_category:
                        x_col = st.selectbox("Select grouping column (categorical):", categorical_cols)
                        if df[x_col].nunique() > 10:
                            st.warning(f"Column {x_col} has {df[x_col].nunique()} unique values, which may make the plot cluttered.")
                
                fig = px.box(
                    df, 
                    x=x_col, 
                    y=y_col,
                    title=f"Box Plot of {y_col}" + (f" grouped by {x_col}" if x_col else "")
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Correlation Heatmap" and len(numerical_cols) > 1:
                # Select columns for correlation
                selected_cols = st.multiselect(
                    "Select columns for correlation matrix:", 
                    numerical_cols,
                    default=numerical_cols[:min(5, len(numerical_cols))]
                )
                
                if selected_cols:
                    corr = df[selected_cols].corr()
                    
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Not enough appropriate columns for the selected visualization type.")
        else:
            st.error("No columns found in the dataset.")
    
    with preprocessing_tab:
        st.subheader("Preprocess Your Data")
        
        # Show current data type information
        st.write("Column Data Types:")
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        }))
        
        # Show unique value counts for categorical columns
        if categorical_cols:
            st.write("Unique Values in Categorical Columns:")
            for col in categorical_cols:
                st.write(f"{col}: {df[col].nunique()} unique values")
        
        # Handling missing values
        st.subheader("Handle Missing Values")
        
        # Missing value statistics
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            cols_with_missing = missing_values[missing_values > 0]
            st.write("Columns with missing values:")
            st.dataframe(pd.DataFrame({
                'Column': cols_with_missing.index,
                'Missing Count': cols_with_missing.values,
                'Missing Percentage': (cols_with_missing.values / len(df) * 100).round(2)
            }))
            
            # Missing value handling options
            missing_strategy = st.selectbox(
                "Select strategy for handling missing values:",
                ["Keep missing values", "Drop rows with missing values", "Fill with mean/mode", "Fill with median", "Fill with a specific value"]
            )
            
            if missing_strategy == "Drop rows with missing values":
                if st.button("Drop rows with missing values"):
                    old_shape = df.shape
                    df = df.dropna()
                    st.success(f"Dropped rows with missing values. Rows before: {old_shape[0]}, Rows after: {df.shape[0]}")
                    datasets['current_df'] = df
            
            elif missing_strategy == "Fill with mean/mode":
                if st.button("Fill missing values with mean/mode"):
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col] = df[col].fillna(df[col].mean())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                    st.success("Filled missing values with mean/mode.")
                    datasets['current_df'] = df
            
            elif missing_strategy == "Fill with median":
                if st.button("Fill missing values with median"):
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                    st.success("Filled missing values with median.")
                    datasets['current_df'] = df
            
            elif missing_strategy == "Fill with a specific value":
                fill_value = st.text_input("Enter value to fill missing data:", "0")
                if st.button("Fill missing values"):
                    df = df.fillna(fill_value)
                    st.success(f"Filled missing values with '{fill_value}'.")
                    datasets['current_df'] = df
        else:
            st.success("No missing values found in the dataset.")
        
        # Feature scaling
        st.subheader("Feature Scaling")
        if numerical_cols:
            scaling_method = st.selectbox(
                "Select scaling method:",
                ["None", "StandardScaler (mean=0, std=1)", "MinMaxScaler (0-1 range)"]
            )
            
            scale_columns = st.multiselect(
                "Select columns to scale:",
                numerical_cols
            )
            
            if scaling_method != "None" and scale_columns and st.button("Apply Scaling"):
                if scaling_method == "StandardScaler (mean=0, std=1)":
                    scaler = StandardScaler()
                    df[scale_columns] = scaler.fit_transform(df[scale_columns])
                    st.success(f"Applied StandardScaler to {len(scale_columns)} columns.")
                
                elif scaling_method == "MinMaxScaler (0-1 range)":
                    # We'll implement MinMaxScaler manually for simplicity
                    for col in scale_columns:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    st.success(f"Applied MinMaxScaler to {len(scale_columns)} columns.")
                
                datasets['current_df'] = df
        
        # Encode categorical variables
        st.subheader("Encode Categorical Variables")
        if categorical_cols:
            encoding_method = st.selectbox(
                "Select encoding method:",
                ["None", "One-Hot Encoding", "Label Encoding"]
            )
            
            encode_columns = st.multiselect(
                "Select categorical columns to encode:",
                categorical_cols
            )
            
            if encoding_method != "None" and encode_columns and st.button("Apply Encoding"):
                if encoding_method == "One-Hot Encoding":
                    old_shape = df.shape
                    df = pd.get_dummies(df, columns=encode_columns, drop_first=False)
                    st.success(f"Applied One-Hot Encoding. Shape before: {old_shape}, Shape after: {df.shape}")
                
                elif encoding_method == "Label Encoding":
                    for col in encode_columns:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                    st.success(f"Applied Label Encoding to {len(encode_columns)} columns.")
                
                datasets['current_df'] = df
        
        # Download processed data
        st.subheader("Download Processed Data")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download processed data as CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

def process_data_for_ml(df, target_column, test_size=0.2, random_state=42):
    """
    Process data for machine learning models
    
    Args:
        df: DataFrame to process
        target_column: Target variable column name
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Processed data splits
    """
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical features
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Apply one-hot encoding to categorical features
    if not cat_cols.empty:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, X.columns
