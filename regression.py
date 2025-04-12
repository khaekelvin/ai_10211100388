import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import data_processing

def render_regression_analysis(datasets):
    st.title("Regression Analysis")
    
    st.markdown("""
    Regression analysis allows you to predict continuous numerical values based on input features.
    This section helps you build and evaluate regression models on your data.
    """)
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ["Use currently loaded dataset", "Upload regression dataset"]
    )
    
    if data_source == "Upload regression dataset":
        uploaded_file = st.file_uploader("Upload CSV file for regression analysis", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                
                # Add to session state for potential future use
                if 'regression_df' not in datasets:
                    datasets['regression_df'] = df
                else:
                    datasets['regression_df'] = df
                    
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                df = None
        else:
            if 'regression_df' in datasets and datasets['regression_df'] is not None:
                df = datasets['regression_df']
                st.info("Using previously uploaded regression dataset")
            else:
                st.info("Please upload a CSV file to continue with regression analysis")
                df = None
    else:
        # Use currently loaded dataset
        if datasets['current_df'] is not None:
            df = datasets['current_df']
            st.success("Using currently loaded dataset")
        else:
            st.error("No dataset is currently loaded. Please upload a dataset or go to the Data Upload section first.")
            return
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Select numerical columns for regression
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Insufficient numerical columns for regression. Need at least 2 numerical columns.")
        return
    
    # Feature and target selection
    st.subheader("Select Features and Target")
    
    target_col = st.selectbox("Select target variable (y):", numeric_cols)
    
    # Filter out target from potential features
    feature_cols = [col for col in df.columns if col != target_col]
    
    selected_features = st.multiselect(
        "Select features (X):",
        feature_cols,
        default=feature_cols[:min(5, len(feature_cols))]
    )
    
    if not selected_features:
        st.warning("Please select at least one feature to proceed.")
        return
    
    # Model selection
    st.subheader("Select Regression Model")
    
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Elastic Net": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Support Vector Regression": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor()
    }
    
    selected_model = st.selectbox("Select regression model:", list(regression_models.keys()))
    
    # Model parameters (simplified for key models)
    st.subheader("Model Parameters")
    
    model_params = {}
    
    if selected_model == "Ridge Regression":
        alpha = st.slider("Alpha (regularization strength):", 0.01, 10.0, 1.0, 0.01)
        model_params['alpha'] = alpha
    
    elif selected_model == "Lasso Regression":
        alpha = st.slider("Alpha (regularization strength):", 0.01, 10.0, 1.0, 0.01)
        model_params['alpha'] = alpha
    
    elif selected_model == "Elastic Net":
        alpha = st.slider("Alpha (regularization strength):", 0.01, 10.0, 1.0, 0.01)
        l1_ratio = st.slider("L1 Ratio (mixing parameter):", 0.0, 1.0, 0.5, 0.01)
        model_params['alpha'] = alpha
        model_params['l1_ratio'] = l1_ratio
    
    elif selected_model == "Decision Tree":
        max_depth = st.slider("Max Depth:", 1, 20, 5)
        min_samples_split = st.slider("Min Samples Split:", 2, 20, 2)
        model_params['max_depth'] = max_depth
        model_params['min_samples_split'] = min_samples_split
    
    elif selected_model == "Random Forest":
        n_estimators = st.slider("Number of Trees:", 10, 200, 100, 10)
        max_depth = st.slider("Max Depth:", 1, 20, 5)
        model_params['n_estimators'] = n_estimators
        model_params['max_depth'] = max_depth
    
    elif selected_model == "Gradient Boosting":
        n_estimators = st.slider("Number of Boosting Stages:", 10, 200, 100, 10)
        learning_rate = st.slider("Learning Rate:", 0.01, 1.0, 0.1, 0.01)
        model_params['n_estimators'] = n_estimators
        model_params['learning_rate'] = learning_rate
    
    elif selected_model == "Support Vector Regression":
        C = st.slider("C (regularization parameter):", 0.1, 10.0, 1.0, 0.1)
        kernel = st.selectbox("Kernel:", ["linear", "poly", "rbf", "sigmoid"])
        model_params['C'] = C
        model_params['kernel'] = kernel
    
    elif selected_model == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors:", 1, 20, 5)
        model_params['n_neighbors'] = n_neighbors
    
    # Training options
    st.subheader("Training Options")
    
    test_size = st.slider("Test Size (%):", 10, 50, 20) / 100
    random_state = st.slider("Random State:", 0, 100, 42)
    
    # Update selected model with parameters
    model = regression_models[selected_model]
    for param, value in model_params.items():
        setattr(model, param, value)
    
    # Prepare and train
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                X = df[selected_features]
                y = df[target_col]
                
                # Data preprocessing
                # Handle categorical features
                cat_cols = X.select_dtypes(include=['object', 'category']).columns
                if not cat_cols.empty:
                    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Scale data if using SVR, KNN, or Elastic Net (models sensitive to feature scaling)
                if selected_model in ["Support Vector Regression", "K-Nearest Neighbors", "Elastic Net"]:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                else:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                
                # Evaluation metrics
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred_test)
                
                train_rmse = np.sqrt(train_mse)
                test_rmse = np.sqrt(test_mse)
                
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Display results
                st.subheader("Model Performance")
                
                # Create two columns for train and test metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Training Data Performance**")
                    st.write(f"R² Score: {train_r2:.4f}")
                    st.write(f"Mean Squared Error: {train_mse:.4f}")
                    st.write(f"Root Mean Squared Error: {train_rmse:.4f}")
                    st.write(f"Mean Absolute Error: {train_mae:.4f}")
                
                with col2:
                    st.markdown("**Testing Data Performance**")
                    st.write(f"R² Score: {test_r2:.4f}")
                    st.write(f"Mean Squared Error: {test_mse:.4f}")
                    st.write(f"Root Mean Squared Error: {test_rmse:.4f}")
                    st.write(f"Mean Absolute Error: {test_mae:.4f}")
                
                # Visualizations
                st.subheader("Model Predictions Visualization")
                
                # Actual vs Predicted Plot
                fig = go.Figure()
                
                # Add scatter plot for actual vs predicted values
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred_test,
                    mode='markers',
                    marker=dict(color='blue'),
                    name='Test Data'
                ))
                
                # Add perfect prediction line (diagonal)
                min_val = min(min(y_test), min(y_pred_test))
                max_val = max(max(y_test), max(y_pred_test))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Prediction'
                ))
                
                fig.update_layout(
                    title='Actual vs Predicted Values',
                    xaxis_title='Actual',
                    yaxis_title='Predicted',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution of Residuals
                residuals = y_test - y_pred_test
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    marker=dict(color='blue'),
                    name='Residuals'
                ))
                
                fig.update_layout(
                    title='Distribution of Residuals',
                    xaxis_title='Residual Value',
                    yaxis_title='Frequency',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance (for applicable models)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    
                    # Get feature importances
                    importances = model.feature_importances_
                    feature_names = X.columns
                    
                    # Create dataframe for feature importance
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    fig = px.bar(
                        feature_importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model coefficients (for linear models)
                elif hasattr(model, 'coef_'):
                    st.subheader("Model Coefficients")
                    
                    # Get coefficients
                    coefficients = model.coef_
                    feature_names = X.columns
                    
                    # Create dataframe for coefficients
                    coef_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefficients
                    }).sort_values('Coefficient', ascending=False)
                    
                    # Plot coefficients
                    fig = px.bar(
                        coef_df,
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        title='Model Coefficients'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Prediction tool
                st.subheader("Make Predictions")
                
                st.markdown("""
                Use this tool to make predictions with your trained model. 
                Enter values for each feature and get a prediction.
                """)
                
                # Create input fields for each feature
                input_data = {}
                
                for feature in X.columns:
                    # Get the min and max values for numerical features
                    if df[feature].dtype in ['int64', 'float64']:
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        step = (max_val - min_val) / 100
                        
                        # Use slider for numerical features
                        input_data[feature] = st.slider(
                            f"{feature}:",
                            min_val, max_val,
                            (min_val + max_val) / 2,  # Default to the middle value
                            step
                        )
                    else:
                        # Use selectbox for categorical features
                        options = df[feature].unique().tolist()
                        input_data[feature] = st.selectbox(f"{feature}:", options)
                
                # Make prediction button
                if st.button("Predict"):
                    try:
                        # Convert input to dataframe
                        input_df = pd.DataFrame([input_data])
                        
                        # Apply the same preprocessing as for training data
                        cat_cols = input_df.select_dtypes(include=['object', 'category']).columns
                        if not cat_cols.empty:
                            input_df = pd.get_dummies(input_df, columns=cat_cols)
                            # Handle missing columns that were in training data
                            for col in X.columns:
                                if col not in input_df.columns:
                                    input_df[col] = 0
                            # Ensure same column order
                            input_df = input_df[X.columns]
                        
                        # Scale input if necessary
                        if selected_model in ["Support Vector Regression", "K-Nearest Neighbors", "Elastic Net"]:
                            input_df = scaler.transform(input_df)
                        
                        # Make prediction
                        prediction = model.predict(input_df)[0]
                        
                        # Display prediction
                        st.success(f"Predicted {target_col}: {prediction:.4f}")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
