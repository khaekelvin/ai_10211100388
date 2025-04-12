import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
import io
import base64
from data_processing import process_data_for_ml

def render_neural_networks(datasets):
    st.title("Neural Networks")
    
    st.markdown("""
    Neural networks are powerful deep learning models that can learn complex patterns in data.
    This section allows you to build, train, and evaluate neural network models for both 
    classification and regression tasks.
    """)
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ["Use currently loaded dataset", "Upload neural network dataset"]
    )
    
    if data_source == "Upload neural network dataset":
        uploaded_file = st.file_uploader("Upload CSV file for neural network analysis", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                
                # Add to session state for potential future use
                if 'nn_df' not in datasets:
                    datasets['nn_df'] = df
                else:
                    datasets['nn_df'] = df
                    
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                df = None
        else:
            if 'nn_df' in datasets and datasets['nn_df'] is not None:
                df = datasets['nn_df']
                st.info("Using previously uploaded neural network dataset")
            else:
                st.info("Please upload a CSV file to continue with neural network analysis")
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
    
    # 1. Define problem type (Classification or Regression)
    st.subheader("Neural Network Type")
    
    problem_type = st.radio(
        "Select problem type:",
        ["Classification", "Regression"]
    )
    
    # 2. Select target variable
    st.subheader("Select Target Variable")
    
    # Determine potential target columns based on problem type
    if problem_type == "Classification":
        potential_targets = df.columns.tolist()
    else:  # Regression
        potential_targets = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not potential_targets:
        st.error(f"No suitable columns found for {problem_type} problem.")
        return
    
    target_column = st.selectbox("Select target variable:", potential_targets)
    
    # 3. Select features
    st.subheader("Select Features")
    
    # Get all columns except the target
    feature_columns = [col for col in df.columns if col != target_column]
    
    if not feature_columns:
        st.error("No feature columns available after selecting target variable.")
        return
    
    selected_features = st.multiselect(
        "Select features:",
        feature_columns,
        default=feature_columns[:min(5, len(feature_columns))]
    )
    
    if not selected_features:
        st.warning("Please select at least one feature to proceed.")
        return
    
    # 4. Configure neural network architecture
    st.subheader("Neural Network Architecture")
    
    # Hidden layers configuration
    st.write("Hidden Layers Configuration")
    
    num_layers = st.slider("Number of hidden layers:", 1, 5, 2)
    
    layers_config = []
    for i in range(num_layers):
        col1, col2 = st.columns(2)
        with col1:
            units = st.slider(f"Units in layer {i+1}:", 1, 256, 64, 8)
        with col2:
            activation = st.selectbox(
                f"Activation for layer {i+1}:",
                ["relu", "sigmoid", "tanh", "linear"],
                index=0
            )
        layers_config.append((units, activation))
        
        # Add option for dropout
        use_dropout = st.checkbox(f"Add dropout after layer {i+1}", value=False)
        if use_dropout:
            dropout_rate = st.slider(f"Dropout rate for layer {i+1}:", 0.0, 0.5, 0.2, 0.05)
            layers_config.append(("dropout", dropout_rate))
    
    # Output layer configuration
    st.write("Output Layer Configuration")
    
    if problem_type == "Regression":
        output_activation = "linear"
        output_units = 1
        st.write("For regression: Using linear activation with 1 output unit")
    else:  # Classification
        # Determine if binary or multi-class
        n_classes = df[target_column].nunique()
        
        if n_classes == 2:
            output_activation = "sigmoid"
            output_units = 1
            st.write(f"Binary classification detected ({n_classes} classes): Using sigmoid activation with 1 output unit")
        else:
            output_activation = "softmax"
            output_units = n_classes
            st.write(f"Multi-class classification detected ({n_classes} classes): Using softmax activation with {n_classes} output units")
    
    # 5. Training parameters
    st.subheader("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.select_slider(
            "Batch size:",
            options=[8, 16, 32, 64, 128, 256],
            value=32
        )
        
        optimizer = st.selectbox(
            "Optimizer:",
            ["adam", "rmsprop", "sgd"],
            index=0
        )
    
    with col2:
        epochs = st.slider("Epochs:", 10, 200, 50, 10)
        
        learning_rate = st.select_slider(
            "Learning rate:",
            options=[0.0001, 0.001, 0.01, 0.1],
            value=0.001
        )
    
    # Use early stopping
    use_early_stopping = st.checkbox("Use Early Stopping", value=True)
    
    if use_early_stopping:
        patience = st.slider("Patience (epochs):", 3, 30, 10)
    
    # Loss function
    if problem_type == "Regression":
        loss_function = st.selectbox(
            "Loss function:",
            ["mean_squared_error", "mean_absolute_error", "huber_loss"],
            index=0
        )
    else:  # Classification
        if n_classes == 2:
            loss_function = "binary_crossentropy"
            st.write("Using binary_crossentropy loss for binary classification")
        else:
            loss_function = "categorical_crossentropy"
            st.write("Using categorical_crossentropy loss for multi-class classification")
    
    # Validation data percentage
    validation_split = st.slider("Validation split:", 0.1, 0.3, 0.2, 0.05)
    
    # Test data percentage 
    test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
    
    # Train model button
    if st.button("Train Neural Network"):
        with st.spinner("Training neural network model..."):
            try:
                # Prepare data
                X = df[selected_features]
                y = df[target_column]
                
                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Process features: handle categorical variables and scale
                cat_features = X.select_dtypes(include=['object', 'category']).columns
                
                if not cat_features.empty:
                    # For categorical features, apply one-hot encoding
                    X_train_encoded = pd.get_dummies(X_train, columns=cat_features)
                    X_test_encoded = pd.get_dummies(X_test, columns=cat_features)
                    
                    # Ensure test set has same columns as train set
                    for col in X_train_encoded.columns:
                        if col not in X_test_encoded.columns:
                            X_test_encoded[col] = 0
                    
                    # Reorder columns to match
                    X_test_encoded = X_test_encoded[X_train_encoded.columns]
                else:
                    X_train_encoded = X_train
                    X_test_encoded = X_test
                
                # Scale numerical features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_encoded)
                X_test_scaled = scaler.transform(X_test_encoded)
                
                # Process targets for classification
                if problem_type == "Classification":
                    if n_classes == 2:  # Binary classification
                        # Convert to 0 and 1
                        le = LabelEncoder()
                        y_train_encoded = le.fit_transform(y_train)
                        y_test_encoded = le.transform(y_test)
                        class_names = le.classes_
                    else:  # Multi-class classification
                        # One-hot encode the labels
                        encoder = OneHotEncoder(sparse_output=False)
                        y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
                        y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))
                        class_names = encoder.categories_[0]
                else:  # Regression
                    y_train_encoded = y_train.values
                    y_test_encoded = y_test.values
                
                # Build neural network model
                model = Sequential()
                
                # Input layer
                first_layer = True
                
                for i, layer_config in enumerate(layers_config):
                    if layer_config[0] == "dropout":
                        model.add(Dropout(layer_config[1]))
                    else:
                        units, activation = layer_config
                        if first_layer:
                            model.add(Dense(units, activation=activation, input_shape=(X_train_scaled.shape[1],)))
                            first_layer = False
                        else:
                            model.add(Dense(units, activation=activation))
                
                # Output layer
                model.add(Dense(output_units, activation=output_activation))
                
                # Configure optimizer
                if optimizer == "adam":
                    opt = Adam(learning_rate=learning_rate)
                elif optimizer == "rmsprop":
                    opt = RMSprop(learning_rate=learning_rate)
                else:  # sgd
                    opt = SGD(learning_rate=learning_rate)
                
                # Compile model
                model.compile(
                    optimizer=opt,
                    loss=loss_function,
                    metrics=['accuracy'] if problem_type == "Classification" else ['mae', 'mse']
                )
                
                # Configure callbacks
                callbacks = []
                if use_early_stopping:
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=patience,
                        restore_best_weights=True
                    )
                    callbacks.append(early_stopping)
                
                # Train model
                history = model.fit(
                    X_train_scaled, y_train_encoded,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Display model summary
                st.subheader("Model Summary")
                
                # Capture model summary
                summary_str = []
                model.summary(print_fn=lambda x: summary_str.append(x))
                summary_str = "\n".join(summary_str)
                
                st.text(summary_str)
                
                # Display training history
                st.subheader("Training History")
                
                # Plot training history
                fig = go.Figure()
                
                # Add loss curves
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(history.history['loss'])+1)),
                    y=history.history['loss'],
                    mode='lines',
                    name='Training Loss'
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(history.history['val_loss'])+1)),
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Validation Loss'
                ))
                
                fig.update_layout(
                    title='Training and Validation Loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    legend=dict(x=0.1, y=0.9),
                    width=700,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # If classification, also plot accuracy
                if problem_type == "Classification":
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(history.history['accuracy'])+1)),
                        y=history.history['accuracy'],
                        mode='lines',
                        name='Training Accuracy'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(history.history['val_accuracy'])+1)),
                        y=history.history['val_accuracy'],
                        mode='lines',
                        name='Validation Accuracy'
                    ))
                    
                    fig.update_layout(
                        title='Training and Validation Accuracy',
                        xaxis_title='Epoch',
                        yaxis_title='Accuracy',
                        legend=dict(x=0.1, y=0.1),
                        width=700,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Evaluate model on test data
                st.subheader("Model Evaluation on Test Data")
                
                # Predictions on test data
                y_pred = model.predict(X_test_scaled)
                
                # Calculate and display metrics
                if problem_type == "Classification":
                    if n_classes == 2:  # Binary classification
                        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                        
                        # Accuracy
                        accuracy = accuracy_score(y_test_encoded, y_pred_classes)
                        st.write(f"Test Accuracy: {accuracy:.4f}")
                        
                        # Classification report
                        report = classification_report(y_test_encoded, y_pred_classes, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.write("Classification Report:")
                        st.dataframe(report_df)
                        
                        # Confusion matrix
                        cm = confusion_matrix(y_test_encoded, y_pred_classes)
                        
                        # Plot confusion matrix
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual"),
                            x=[f"{class_names[0]}", f"{class_names[1]}"],
                            y=[f"{class_names[0]}", f"{class_names[1]}"],
                            text_auto=True,
                            color_continuous_scale="Blues"
                        )
                        
                        fig.update_layout(title="Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:  # Multi-class classification
                        y_pred_classes = np.argmax(y_pred, axis=1)
                        y_true_classes = np.argmax(y_test_encoded, axis=1)
                        
                        # Accuracy
                        accuracy = accuracy_score(y_true_classes, y_pred_classes)
                        st.write(f"Test Accuracy: {accuracy:.4f}")
                        
                        # Classification report
                        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.write("Classification Report:")
                        st.dataframe(report_df)
                        
                        # Confusion matrix
                        cm = confusion_matrix(y_true_classes, y_pred_classes)
                        
                        # Plot confusion matrix
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual"),
                            x=[f"{c}" for c in class_names],
                            y=[f"{c}" for c in class_names],
                            text_auto=True,
                            color_continuous_scale="Blues"
                        )
                        
                        fig.update_layout(title="Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                
                else:  # Regression
                    # Calculate regression metrics
                    mse = mean_squared_error(y_test_encoded, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test_encoded, y_pred)
                    r2 = r2_score(y_test_encoded, y_pred)
                    
                    # Display metrics
                    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                    st.write(f"R² Score: {r2:.4f}")
                    
                    # Plot actual vs predicted
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=y_test_encoded,
                        y=y_pred.flatten(),
                        mode='markers',
                        name='Predictions'
                    ))
                    
                    # Add perfect prediction line
                    min_val = min(min(y_test_encoded), min(y_pred.flatten()))
                    max_val = max(max(y_test_encoded), max(y_pred.flatten()))
                    
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Actual vs Predicted Values',
                        xaxis_title='Actual',
                        yaxis_title='Predicted',
                        legend=dict(x=0.1, y=0.9),
                        width=700,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Make predictions with trained model
                st.subheader("Make Predictions")
                
                st.write("Use the trained model to make predictions on new data")
                
                # Create input form with sliders for numeric features and selectboxes for categorical
                input_values = {}
                
                for feature in selected_features:
                    if df[feature].dtype in ['int64', 'float64']:
                        # Numeric feature - use slider
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        step = (max_val - min_val) / 100
                        
                        input_values[feature] = st.slider(
                            f"{feature}:",
                            min_val, max_val,
                            (min_val + max_val) / 2,  # Default to the middle value
                            step
                        )
                    else:
                        # Categorical feature - use selectbox
                        options = df[feature].unique().tolist()
                        input_values[feature] = st.selectbox(f"{feature}:", options)
                
                # Make prediction
                if st.button("Predict"):
                    # Create a dataframe from input
                    input_df = pd.DataFrame([input_values])
                    
                    # Process the input data similar to training data
                    if not cat_features.empty:
                        input_df_encoded = pd.get_dummies(input_df, columns=cat_features)
                        
                        # Add missing columns
                        for col in X_train_encoded.columns:
                            if col not in input_df_encoded.columns:
                                input_df_encoded[col] = 0
                                
                        # Ensure same column order
                        input_df_encoded = input_df_encoded[X_train_encoded.columns]
                    else:
                        input_df_encoded = input_df
                    
                    # Scale the input data
                    input_scaled = scaler.transform(input_df_encoded)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    
                    # Display prediction
                    if problem_type == "Classification":
                        if n_classes == 2:
                            pred_class = class_names[1] if prediction[0][0] > 0.5 else class_names[0]
                            confidence = max(prediction[0][0], 1 - prediction[0][0])
                            st.success(f"Predicted class: {pred_class} (confidence: {confidence:.4f})")
                        else:
                            pred_idx = np.argmax(prediction[0])
                            pred_class = class_names[pred_idx]
                            confidence = prediction[0][pred_idx]
                            st.success(f"Predicted class: {pred_class} (confidence: {confidence:.4f})")
                            
                            # Show full probabilities
                            st.write("Class probabilities:")
                            probs_df = pd.DataFrame({
                                'Class': class_names,
                                'Probability': prediction[0]
                            })
                            st.dataframe(probs_df.sort_values('Probability', ascending=False))
                    else:
                        st.success(f"Predicted value: {prediction[0][0]:.4f}")
                
            except Exception as e:
                st.error(f"Error during neural network training: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
