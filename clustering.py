import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import data_processing

def render_clustering_analysis(datasets):
    st.title("Clustering Analysis")
    
    st.markdown("""
    Clustering helps identify natural groupings in data. This section lets you apply various
    clustering algorithms to discover patterns and segments in your data.
    """)
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ["Use currently loaded dataset", "Upload clustering dataset"]
    )
    
    if data_source == "Upload clustering dataset":
        uploaded_file = st.file_uploader("Upload CSV file for clustering analysis", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                
                # Add to session state for potential future use
                if 'clustering_df' not in datasets:
                    datasets['clustering_df'] = df
                else:
                    datasets['clustering_df'] = df
                    
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                df = None
        else:
            if 'clustering_df' in datasets and datasets['clustering_df'] is not None:
                df = datasets['clustering_df']
                st.info("Using previously uploaded clustering dataset")
            else:
                st.info("Please upload a CSV file to continue with clustering analysis")
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
    
    # Feature selection
    st.subheader("Select Features for Clustering")
    
    # Get numerical columns for clustering
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Insufficient numerical columns for clustering. Need at least 2 numerical columns.")
        return
    
    selected_features = st.multiselect(
        "Select features for clustering:",
        numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))]
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features to proceed with clustering.")
        return
    
    # Preprocessing
    st.subheader("Data Preprocessing")
    
    scale_data = st.checkbox("Standardize Features (recommended)", value=True)
    apply_pca = st.checkbox("Apply PCA for Dimensionality Reduction")
    
    n_components = 2
    if apply_pca:
        max_components = min(len(selected_features), 10)
        n_components = st.slider("Number of PCA Components", 2, max_components, 2)
    
    # Clustering algorithms
    st.subheader("Clustering Algorithm")
    
    clustering_algs = {
        "K-Means": "KMeans",
        "Hierarchical Clustering": "Hierarchical",
        "DBSCAN": "DBSCAN",
        "Gaussian Mixture Model": "GMM"
    }
    
    selected_alg = st.selectbox("Select Clustering Algorithm:", list(clustering_algs.keys()))
    
    # Algorithm parameters
    st.subheader("Algorithm Parameters")
    
    if selected_alg == "K-Means":
        n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
        kmeans_params = {
            'n_clusters': n_clusters,
            'random_state': 42,
            'n_init': 10
        }
    
    elif selected_alg == "Hierarchical Clustering":
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        linkage = st.selectbox("Linkage Criterion", ["ward", "complete", "average", "single"])
        hierarchical_params = {
            'n_clusters': n_clusters,
            'linkage': linkage
        }
    
    elif selected_alg == "DBSCAN":
        eps = st.slider("EPS (Maximum Distance Between Points)", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("Min Samples", 2, 20, 5)
        dbscan_params = {
            'eps': eps,
            'min_samples': min_samples
        }
    
    elif selected_alg == "Gaussian Mixture Model":
        n_components_gmm = st.slider("Number of Components", 2, 10, 3)
        covariance_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
        gmm_params = {
            'n_components': n_components_gmm,
            'covariance_type': covariance_type,
            'random_state': 42
        }
    
    # Run clustering
    if st.button("Run Clustering"):
        with st.spinner("Performing clustering..."):
            try:
                # Prepare data
                X = df[selected_features].copy()
                
                # Handle missing values
                X = X.fillna(X.mean())
                
                # Standardize if selected
                if scale_data:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X.values
                
                # Apply PCA if selected
                if apply_pca:
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Display explained variance
                    explained_variance = pca.explained_variance_ratio_
                    total_variance = sum(explained_variance)
                    
                    st.write(f"Total Explained Variance: {total_variance:.4f} ({total_variance*100:.2f}%)")
                    
                    # Display individual component variances
                    for i, var in enumerate(explained_variance):
                        st.write(f"Component {i+1}: {var:.4f} ({var*100:.2f}%)")
                    
                    # Use PCA results for clustering
                    X_for_clustering = X_pca
                    feature_names = [f"PC{i+1}" for i in range(n_components)]
                else:
                    X_for_clustering = X_scaled
                    feature_names = selected_features
                
                # Apply clustering algorithm
                if selected_alg == "K-Means":
                    model = KMeans(**kmeans_params)
                    labels = model.fit_predict(X_for_clustering)
                    
                    # Evaluate with silhouette score
                    silhouette_avg = silhouette_score(X_for_clustering, labels)
                    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                    
                    # Store cluster centers
                    if apply_pca:
                        centers = model.cluster_centers_
                    else:
                        centers = model.cluster_centers_
                
                elif selected_alg == "Hierarchical Clustering":
                    model = AgglomerativeClustering(**hierarchical_params)
                    labels = model.fit_predict(X_for_clustering)
                    
                    # Evaluate with silhouette score
                    silhouette_avg = silhouette_score(X_for_clustering, labels)
                    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                    
                    # No centers for hierarchical clustering
                    centers = None
                
                elif selected_alg == "DBSCAN":
                    model = DBSCAN(**dbscan_params)
                    labels = model.fit_predict(X_for_clustering)
                    
                    # Count number of clusters (excluding noise points with label -1)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    st.write(f"Number of clusters found: {n_clusters}")
                    
                    # Count noise points
                    n_noise = list(labels).count(-1)
                    st.write(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
                    
                    if len(set(labels)) > 1:  # If there's more than just noise points
                        silhouette_avg = silhouette_score(X_for_clustering, labels)
                        st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                    
                    # No centers for DBSCAN
                    centers = None
                
                elif selected_alg == "Gaussian Mixture Model":
                    model = GaussianMixture(**gmm_params)
                    model.fit(X_for_clustering)
                    labels = model.predict(X_for_clustering)
                    
                    # Evaluate with silhouette score
                    silhouette_avg = silhouette_score(X_for_clustering, labels)
                    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                    
                    # Store means as centers
                    centers = model.means_
                
                # Add cluster labels to dataframe
                result_df = df.copy()
                result_df['Cluster'] = labels
                
                # Display cluster statistics
                st.subheader("Cluster Statistics")
                
                # Count observations in each cluster
                cluster_counts = result_df['Cluster'].value_counts().sort_index()
                
                # If there are noise points (DBSCAN), label them properly
                if -1 in cluster_counts.index:
                    cluster_counts = cluster_counts.rename({-1: 'Noise'})
                
                st.write("Observations per cluster:")
                st.write(cluster_counts)
                
                # Show statistics per cluster for selected features
                st.write("Feature statistics by cluster:")
                
                cluster_stats = result_df.groupby('Cluster')[selected_features].agg(
                    ['mean', 'std', 'min', 'max']
                )
                st.dataframe(cluster_stats)
                
                # Visualizations
                st.subheader("Cluster Visualizations")
                
                # Choose which features/components to plot
                if apply_pca:
                    # PCA visualization
                    if n_components >= 2:
                        fig = px.scatter(
                            x=X_pca[:, 0],
                            y=X_pca[:, 1],
                            color=labels,
                            labels={'x': 'PC1', 'y': 'PC2'},
                            title='Cluster Visualization (PCA)',
                            color_continuous_scale=px.colors.qualitative.G10
                        )
                        
                        # Add cluster centers if available
                        if centers is not None:
                            fig.add_trace(go.Scatter(
                                x=centers[:, 0],
                                y=centers[:, 1],
                                mode='markers',
                                marker=dict(
                                    color='black',
                                    size=10,
                                    symbol='x'
                                ),
                                name='Cluster Centers'
                            ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # If we have 3 or more PCA components, show 3D plot
                        if n_components >= 3:
                            fig = px.scatter_3d(
                                x=X_pca[:, 0],
                                y=X_pca[:, 1],
                                z=X_pca[:, 2],
                                color=labels,
                                labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                                title='3D Cluster Visualization (PCA)',
                                color_continuous_scale=px.colors.qualitative.G10
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # Feature visualization
                    if len(selected_features) >= 2:
                        # Allow user to select which features to visualize
                        viz_x = st.selectbox("X-axis feature:", selected_features, index=0)
                        viz_y = st.selectbox("Y-axis feature:", 
                                            [f for f in selected_features if f != viz_x], 
                                            index=0 if selected_features[0] != viz_x else 1)
                        
                        fig = px.scatter(
                            result_df,
                            x=viz_x,
                            y=viz_y,
                            color='Cluster',
                            title=f'Cluster Visualization ({viz_x} vs {viz_y})',
                            color_continuous_scale=px.colors.qualitative.G10
                        )
                        
                        # Add cluster centers if available (K-Means or GMM)
                        if centers is not None:
                            # Get indices of selected features
                            idx_x = selected_features.index(viz_x)
                            idx_y = selected_features.index(viz_y)
                            
                            fig.add_trace(go.Scatter(
                                x=centers[:, idx_x],
                                y=centers[:, idx_y],
                                mode='markers',
                                marker=dict(
                                    color='black',
                                    size=10,
                                    symbol='x'
                                ),
                                name='Cluster Centers'
                            ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 3D visualization if we have 3 or more features
                        if len(selected_features) >= 3:
                            viz_z = st.selectbox("Z-axis feature:", 
                                                [f for f in selected_features if f != viz_x and f != viz_y])
                            
                            fig = px.scatter_3d(
                                result_df,
                                x=viz_x,
                                y=viz_y,
                                z=viz_z,
                                color='Cluster',
                                title=f'3D Cluster Visualization ({viz_x}, {viz_y}, {viz_z})',
                                color_continuous_scale=px.colors.qualitative.G10
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.subheader("Download Clustering Results")
                
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download clustered data as CSV",
                    data=csv,
                    file_name="clustering_results.csv",
                    mime="text/csv"
                )
                
                # Optional: Cluster interpretation
                st.subheader("Cluster Interpretation")
                
                for cluster in sorted(result_df['Cluster'].unique()):
                    cluster_name = "Noise Points" if cluster == -1 else f"Cluster {cluster}"
                    
                    with st.expander(f"{cluster_name} ({cluster_counts.get(cluster, 0)} observations)"):
                        cluster_df = result_df[result_df['Cluster'] == cluster]
                        
                        # Get statistics for this cluster
                        stats = pd.DataFrame({
                            'Mean': cluster_df[selected_features].mean(),
                            'Overall Mean': df[selected_features].mean(),
                            'Difference (%)': ((cluster_df[selected_features].mean() - df[selected_features].mean()) / df[selected_features].mean() * 100).round(2)
                        }).sort_values('Difference (%)', ascending=False)
                        
                        st.dataframe(stats)
                        
                        st.write("Key characteristics:")
                        for feature in selected_features:
                            mean_diff_pct = ((cluster_df[feature].mean() - df[feature].mean()) / df[feature].mean() * 100)
                            
                            if abs(mean_diff_pct) > 10:  # Only show significant differences
                                direction = "higher" if mean_diff_pct > 0 else "lower"
                                st.write(f"• {feature}: {abs(mean_diff_pct):.1f}% {direction} than average")
            
            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")
