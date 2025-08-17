"""
Catchment Analysis - Updated Version

This script analyzes the influence of geomorphological and land use factors on river systems
and their evolution over time (1959, 1974, 2020) in the North Tanganyika Rift region.

Key analyses:
1. Correlation between geomorphological/land use variables and river characteristics
2. Linear regression and Random Forest modeling of river characteristics
3. Analysis of sinuosity change categories (Increased, Stable, Decreased)
4. Temporal analysis of river systems
5. Catchment clustering and typology analysis
6. Spatial distribution of river characteristics
7. Stream order-based analysis

Modified: 2025-07-08 14:49:50

"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
import os
import warnings
from math import pi

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ====================================
# INPUT VARIABLES
# ====================================

# Set output directory for all figures and results
data_path = r""
output_dir = r""


# ====================================
# UTILITY FUNCTIONS
# ====================================

def save_figure(filename):
    """Create full path for saving figures"""
    return os.path.join(output_dir, filename)

def compare_rf_linear(X, y, feature_names, target_name, output_dir):
    """
    Compare Random Forest and Linear Regression models for a continuous outcome variable

    Parameters:
    -----------
    X : DataFrame or array
        Predictor variables
    y : Series or array
        Target variable (continuous)
    feature_names : list
        Names of features
    target_name : str
        Name of target variable
    output_dir : str
        Directory to save outputs
    """
    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Compare model performance using cross-validation
    print(f"\nComparing models for predicting {target_name}...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        # Use cross-validation to get more reliable estimates
        cv_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))

        results[name] = {
            'R2 (mean)': cv_r2.mean(),
            'R2 (std)': cv_r2.std(),
            'RMSE (mean)': cv_rmse.mean(),
            'RMSE (std)': cv_rmse.std()
        }

        print(f"  {name}: R² = {cv_r2.mean():.4f} (±{cv_r2.std():.4f}), "
              f"RMSE = {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")

    # Convert results to DataFrame for easier display
    results_df = pd.DataFrame(results).T

    # Create comparison plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x=results_df.index, y='R2 (mean)', data=results_df)
    plt.title(f'Model Comparison - R² for {target_name}')
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    sns.barplot(x=results_df.index, y='RMSE (mean)', data=results_df)
    plt.title(f'Model Comparison - RMSE for {target_name}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'model_comparison_{target_name}.png'), dpi=300)
    print(f"Saved model comparison to '{os.path.join(output_dir, f'model_comparison_{target_name}.png')}'")

    # Train best model on full dataset and analyze feature importance
    best_model_name = results_df['R2 (mean)'].idxmax()
    print(f"\nTraining best model ({best_model_name}) on full dataset...")

    best_model = models[best_model_name].fit(X, y)

    # Get feature importance
    if best_model_name == 'Random Forest':
        importance = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title(f'Top 15 Features for {target_name} - Random Forest')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_importance_rf_{target_name}.png'), dpi=300)
        print(f"Saved feature importance to '{os.path.join(output_dir, f'feature_importance_rf_{target_name}.png')}'")
    else:
        # For linear regression
        coefficients = best_model.coef_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', key=abs, ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(15))
        plt.title(f'Top 15 Features for {target_name} - Linear Regression')
        plt.axvline(x=0, color='r', linestyle='-')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_importance_linear_{target_name}.png'), dpi=300)
        print(f"Saved feature importance to '{os.path.join(output_dir, f'feature_importance_linear_{target_name}.png')}'")

    return best_model, feature_importance

def analyze_temporal_trends(catchments, output_dir):
    """
    Analyze temporal trends in river characteristics and forest cover across the catchments

    Parameters:
    -----------
    catchments : GeoDataFrame
        The catchment data with temporal information
    output_dir : str
        Directory to save outputs
    """
    print("\nAnalyzing temporal trends in river characteristics and forest cover...")

    # Create a long-format dataframe for temporal analysis
    # First, create a dataframe for sinuosity
    sin_data = pd.DataFrame({
        'Catchment_ID': np.repeat(catchments['ID'].values, 3),
        'Year': np.tile(['1959', '1974', '2020'], len(catchments)),
        'Sinuosity': np.concatenate([
            catchments['57_59_Sin_'].astype(float).values,
            catchments['73_74_Sin_'].astype(float).values,
            catchments['Google_Sin'].astype(float).values
        ])
    })

    # Similarly for delta presence - ensure it's numeric
    delta_data = pd.DataFrame({
        'Catchment_ID': np.repeat(catchments['ID'].values, 3),
        'Year': np.tile(['1959', '1974', '2020'], len(catchments)),
        'Delta_Present': np.concatenate([
            catchments['57_59_Delt'].astype(float).values,
            catchments['73_74_Delt'].astype(float).values,
            catchments['Google_Del'].astype(float).values
        ])
    })

    # For dams presence - ensure it's numeric
    dam_data = pd.DataFrame({
        'Catchment_ID': np.repeat(catchments['ID'].values, 3),
        'Year': np.tile(['1959', '1974', '2020'], len(catchments)),
        'Dam_Present': np.concatenate([
            catchments['57_59_Bar_'].astype(float).values,
            catchments['73_74_Bar_'].astype(float).values,
            catchments['Google_Bar'].astype(float).values
        ])
    })

    # For forest cover if available (but we don't analyze it separately)
    if all(col in catchments.columns for col in ['forest_59_pct', 'forest_74_pct', 'forest_20_pct']):
        forest_data = pd.DataFrame({
            'Catchment_ID': np.repeat(catchments['ID'].values, 3),
            'Year': np.tile(['1959', '1974', '2020'], len(catchments)),
            'Forest_Cover': np.concatenate([
                catchments['forest_59_pct'].astype(float).values,
                catchments['forest_74_pct'].astype(float).values,
                catchments['forest_20_pct'].astype(float).values
            ])
        })
    else:
        forest_data = None

    # 1. Overall temporal trends
    print("Analyzing overall temporal trends...")

    # Sinuosity trends
    plt.figure(figsize=(12, 8))

    # Use boxplot instead of boxplot to fix scaling issues
    sns.boxplot(data=catchments[['57_59_Sin_', '73_74_Sin_', 'Google_Sin']])
    plt.title('River Sinuosity Changes Over Time')
    plt.ylabel('Sinuosity')
    plt.xlabel('Time Period')
    plt.xticks([0, 1, 2], ['1957-1959', '1973-1974', '2020'])
    plt.savefig(save_figure('sinuosity_temporal_changes.png'), dpi=300)
    print(f"Saved temporal changes plot to '{save_figure('sinuosity_temporal_changes.png')}'")

    # Delta and dam presence over time
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    delta_means = delta_data.groupby('Year')['Delta_Present'].mean()
    sns.barplot(x=delta_means.index, y=delta_means.values)
    plt.title('Proportion of Catchments with Deltas')
    plt.ylabel('Proportion')
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    dam_means = dam_data.groupby('Year')['Dam_Present'].mean()
    sns.barplot(x=dam_means.index, y=dam_means.values)
    plt.title('Proportion of Catchments with Dams')
    plt.ylabel('Proportion')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_dam_temporal_trends.png'), dpi=300)
    print(f"Saved delta and dam trends to '{os.path.join(output_dir, 'delta_dam_temporal_trends.png')}'")

    # 2. Sinuosity trends by catchment characteristics
    print("Analyzing sinuosity trends by catchment characteristics...")

    # Add catchment characteristics to the sinuosity data
    try:
        # Create index mapping for catchment IDs to simplify indexing
        catchment_id_index = {id_val: i for i, id_val in enumerate(catchments['ID'])}

        # Map catchment IDs to indices for faster lookup
        sin_indices = [catchment_id_index.get(cid, 0) for cid in sin_data['Catchment_ID']]

        # Use the indices to get slope values
        sin_data['slope_category'] = pd.cut(
            catchments.iloc[sin_indices]['slope_mean'].values,
            bins=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )

        # Create forest cover category
        sin_data['forest_category'] = pd.cut(
            catchments.iloc[sin_indices]['pct_foret'].values,
            bins=[0, 25, 50, 75, 100], labels=['Very Low', 'Low', 'Medium', 'High']
        )

        # Add stream order if available
        if 'stream_order' in catchments.columns:
            sin_data['stream_order'] = catchments.iloc[sin_indices]['stream_order'].values

        # Plot sinuosity trends by slope and forest categories
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        sns.lineplot(data=sin_data, x='Year', y='Sinuosity', hue='slope_category', marker='o')
        plt.title('Sinuosity Trends by Slope Category')
        plt.ylabel('Average Sinuosity')

        plt.subplot(2, 1, 2)
        sns.lineplot(data=sin_data, x='Year', y='Sinuosity', hue='forest_category', marker='o')
        plt.title('Sinuosity Trends by Forest Cover')
        plt.ylabel('Average Sinuosity')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sinuosity_trends_by_category.png'), dpi=300)
        print(f"Saved sinuosity trends by category to '{os.path.join(output_dir, 'sinuosity_trends_by_category.png')}'")

        # If stream order is available, plot sinuosity trends by stream order
        if 'stream_order' in catchments.columns:
            # Group stream orders into meaningful categories
            sin_data['order_category'] = pd.cut(
                sin_data['stream_order'],
                bins=[0, 1, 2, float('inf')],
                labels=['Order 1', 'Order 2', 'Order 3+']
            )

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=sin_data, x='Year', y='Sinuosity', hue='order_category', marker='o')
            plt.title('Sinuosity Trends by Stream Order')
            plt.ylabel('Average Sinuosity')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sinuosity_trends_by_stream_order.png'), dpi=300)
            print(f"Saved sinuosity trends by stream order to '{os.path.join(output_dir, 'sinuosity_trends_by_stream_order.png')}'")
    except Exception as e:
        print(f"Could not create categorical trend plots: {e}")

    # 3. Check for significant changes between time periods
    print("Testing for significant changes between time periods...")

    # Sinuosity changes
    sin_1959 = catchments['57_59_Sin_'].astype(float)
    sin_1974 = catchments['73_74_Sin_'].astype(float)
    sin_2020 = catchments['Google_Sin'].astype(float)

    # Paired t-test for sinuosity changes
    try:
        t_5974, p_5974 = stats.ttest_rel(sin_1959, sin_1974, nan_policy='omit')
        t_7420, p_7420 = stats.ttest_rel(sin_1974, sin_2020, nan_policy='omit')
        t_5920, p_5920 = stats.ttest_rel(sin_1959, sin_2020, nan_policy='omit')

        print(f"\nSinuosity Changes - Statistical Tests:")
        print(f"1959 to 1974: t={t_5974:.3f}, p={p_5974:.4f} {'(significant)' if p_5974 < 0.05 else ''}")
        print(f"1974 to 2020: t={t_7420:.3f}, p={p_7420:.4f} {'(significant)' if p_7420 < 0.05 else ''}")
        print(f"1959 to 2020: t={t_5920:.3f}, p={p_5920:.4f} {'(significant)' if p_5920 < 0.05 else ''}")
    except Exception as e:
        print(f"Could not perform t-tests for sinuosity: {e}")

    # McNemar's test for delta appearance/disappearance
    try:
        # For deltas - convert to boolean to ensure proper data type
        delta_1959 = catchments['57_59_Delt'].astype(bool)
        delta_1974 = catchments['73_74_Delt'].astype(bool)
        delta_2020 = catchments['Google_Del'].astype(bool)

        # Create contingency tables
        delta_5974 = pd.crosstab(delta_1959, delta_1974)
        delta_7420 = pd.crosstab(delta_1974, delta_2020)
        delta_5920 = pd.crosstab(delta_1959, delta_2020)

        # McNemar's test
        mcnemar_5974 = mcnemar(delta_5974, exact=True)
        mcnemar_7420 = mcnemar(delta_7420, exact=True)
        mcnemar_5920 = mcnemar(delta_5920, exact=True)

        print(f"\nDelta Presence Changes - Statistical Tests:")
        print(f"1959 to 1974: statistic={mcnemar_5974.statistic:.3f}, p={mcnemar_5974.pvalue:.4f} {'(significant)' if mcnemar_5974.pvalue < 0.05 else ''}")
        print(f"1974 to 2020: statistic={mcnemar_7420.statistic:.3f}, p={mcnemar_7420.pvalue:.4f} {'(significant)' if mcnemar_7420.pvalue < 0.05 else ''}")
        print(f"1959 to 2020: statistic={mcnemar_5920.statistic:.3f}, p={mcnemar_5920.pvalue:.4f} {'(significant)' if mcnemar_5920.pvalue < 0.05 else ''}")
    except Exception as e:
        print(f"Could not perform McNemar's test on delta data: {e}")

    # Similar test for dams
    try:
        dam_1959 = catchments['57_59_Bar_'].astype(bool)
        dam_1974 = catchments['73_74_Bar_'].astype(bool)
        dam_2020 = catchments['Google_Bar'].astype(bool)

        # Create contingency tables
        dam_5974 = pd.crosstab(dam_1959, dam_1974)
        dam_7420 = pd.crosstab(dam_1974, dam_2020)
        dam_5920 = pd.crosstab(dam_1959, dam_2020)

        # McNemar's test
        mcnemar_5974 = mcnemar(dam_5974, exact=True)
        mcnemar_7420 = mcnemar(dam_7420, exact=True)
        mcnemar_5920 = mcnemar(dam_5920, exact=True)

        print(f"\nDam Presence Changes - Statistical Tests:")
        print(f"1959 to 1974: statistic={mcnemar_5974.statistic:.3f}, p={mcnemar_5974.pvalue:.4f} {'(significant)' if mcnemar_5974.pvalue < 0.05 else ''}")
        print(f"1974 to 2020: statistic={mcnemar_7420.statistic:.3f}, p={mcnemar_7420.pvalue:.4f} {'(significant)' if mcnemar_7420.pvalue < 0.05 else ''}")
        print(f"1959 to 2020: statistic={mcnemar_5920.statistic:.3f}, p={mcnemar_5920.pvalue:.4f} {'(significant)' if mcnemar_5920.pvalue < 0.05 else ''}")
    except Exception as e:
        print(f"Could not perform McNemar's test on dam data: {e}")

    return sin_data, delta_data, dam_data

def cluster_catchments(catchments, geomorpho_vars, landuse_vars, output_dir):
    """
    Cluster catchments based on their geomorphological and land use characteristics

    Parameters:
    -----------
    catchments : GeoDataFrame
        The catchment data
    geomorpho_vars : list
        List of geomorphological variables
    landuse_vars : list
        List of land use variables
    output_dir : str
        Directory to save outputs
    """
    print("\nPerforming catchment clustering analysis...")

    # Prepare data for clustering
    print("Preparing data for clustering...")
    cluster_vars = geomorpho_vars.copy() + landuse_vars.copy()

    # Remove 'litho_majo' if it's categorical
    if 'litho_majo' in cluster_vars and catchments['litho_majo'].dtype == 'object':
        cluster_vars.remove('litho_majo')
        print("Removed categorical variable 'litho_majo' from clustering analysis")

    # Select variables and handle missing values
    X = catchments[cluster_vars].copy()
    X = X.fillna(X.mean())

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal number of clusters using silhouette score
    print("Determining optimal number of clusters...")
    silhouette_scores = []
    K = range(2, min(11, len(catchments) // 5))  # Ensure we don't have too many clusters for the data

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"  k={k}: silhouette score = {score:.4f}")

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'cluster_silhouette_scores.png'), dpi=300)
    print(f"Saved silhouette scores plot to '{os.path.join(output_dir, 'cluster_silhouette_scores.png')}'")

    # Get optimal k
    optimal_k = K[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")

    # Apply K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster labels to the dataframe
    catchments['cluster'] = clusters
    print(f"Assigned {optimal_k} cluster labels to catchments")

    # Apply PCA for visualization
    print("Applying PCA for cluster visualization...")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Create a dataframe with principal components and cluster labels
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=['PC1', 'PC2']
    )
    pca_df['Cluster'] = clusters

    # Visualize clusters in PCA space
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100)
    plt.title('Catchment Clusters in PCA Space')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'catchment_clusters_pca.png'), dpi=300)
    print(f"Saved cluster visualization to '{os.path.join(output_dir, 'catchment_clusters_pca.png')}'")

    # Analyze characteristics of each cluster
    print("Analyzing characteristics of each cluster...")

    # Include river characteristic variables if available
    analysis_vars = cluster_vars.copy()
    for var in ['Google_Sin', 'sin_change_5920', 'delta_appear_7420']:
        if var in catchments.columns:
            analysis_vars.append(var)

    cluster_stats = catchments.groupby('cluster')[analysis_vars].mean()

    # Create radar charts for each cluster
    # Select a subset of important variables for the radar chart
    radar_vars = ['slope_mean', 'alt_mean', 'TWI_mean', 'KSN_mean', 'pct_foret', 'pct_cult', 'def_2014_pct']

    # Filter to include only variables that exist in the data
    radar_vars = [var for var in radar_vars if var in cluster_stats.columns]

    if len(radar_vars) >= 3:  # Need at least 3 variables for a radar chart
        try:
            # Normalize the variables for radar chart
            radar_data = cluster_stats[radar_vars].copy()
            for var in radar_vars:
                radar_data[var] = (radar_data[var] - radar_data[var].min()) / (radar_data[var].max() - radar_data[var].min() + 1e-10)

            # Number of variables
            N = len(radar_vars)

            # Create angle for each variable
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Close the loop

            # Create radar chart
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

            # Add variable labels
            plt.xticks(angles[:-1], radar_vars, size=12)

            # Draw one axis per variable and add labels
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
            plt.ylim(0, 1)

            # Plot each cluster
            for i in range(optimal_k):
                # Get values for this cluster
                values = radar_data.iloc[i].values.tolist()

                # Make sure values has the same length as angles by adding the first value at the end
                values += [values[0]]  # This ensures values is the same length as angles

                # Debug check - print dimensions
                print(f"  Cluster {i}: angles length = {len(angles)}, values length = {len(values)}")

                # Plot the data
                if len(values) == len(angles):  # Double check dimensions match
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {i}")
                    ax.fill(angles, values, alpha=0.1)

            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Characteristics of Catchment Clusters', size=15)
            plt.savefig(os.path.join(output_dir, 'cluster_radar_chart.png'), dpi=300)
            print(f"Saved cluster radar chart to '{os.path.join(output_dir, 'cluster_radar_chart.png')}'")

        except Exception as e:
            print(f"Error creating radar chart: {e}")
            print("Skipping radar chart visualization")
    else:
        print("Not enough variables for radar chart visualization")

    # Create map of clusters
    plt.figure(figsize=(15, 10))
    catchments.plot(column='cluster', categorical=True, legend=True, cmap='viridis')
    plt.title('Spatial Distribution of Catchment Clusters')
    plt.savefig(os.path.join(output_dir, 'cluster_spatial_map.png'), dpi=300)
    print(f"Saved spatial cluster map to '{os.path.join(output_dir, 'cluster_spatial_map.png')}'")

    # Compare river characteristics between clusters - check if columns exist first
    try:
        plt.figure(figsize=(15, 5))

        # Current sinuosity
        if 'Google_Sin' in catchments.columns:
            plt.subplot(1, 3, 1)
            sns.boxplot(x='cluster', y='Google_Sin', data=catchments)
            plt.title('Current Sinuosity by Cluster')

        # Sinuosity change category
        if 'sin_change_cat_5920' in catchments.columns:
            plt.subplot(1, 3, 2)
            # Convert categorical to numeric for boxplot
            cat_map = {'Decreased': -1, 'Stable': 0, 'Increased': 1}
            temp_data = catchments.copy()
            if temp_data['sin_change_cat_5920'].dtype == 'object' or temp_data['sin_change_cat_5920'].dtype.name == 'category':
                temp_data['sin_change_cat_num'] = temp_data['sin_change_cat_5920'].map(cat_map)
                sns.boxplot(x='cluster', y='sin_change_cat_num', data=temp_data)
                plt.yticks([-1, 0, 1], ['Decreased', 'Stable', 'Increased'])
            else:
                sns.boxplot(x='cluster', y='sin_change_cat_5920', data=catchments)
            plt.title('Sinuosity Change Category (1959-2020) by Cluster')

        # Delta appearance
        if 'delta_appear_7420' in catchments.columns:
            plt.subplot(1, 3, 3)
            sns.barplot(x='cluster', y='delta_appear_7420', data=catchments)
            plt.title('Delta Appearance Rate (1974-2020) by Cluster')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_characteristics.png'), dpi=300)
        print(f"Saved cluster characteristics to '{os.path.join(output_dir, 'cluster_characteristics.png')}'")
    except Exception as e:
        print(f"Error creating cluster characteristics plot: {e}")
        print("Skipping cluster characteristics visualization")

    return catchments, cluster_stats


def analyze_by_stream_order(catchments, geomorpho_vars, landuse_vars, output_dir):
    """
    Analyze relationships between variables by stream order

    Parameters:
    -----------
    catchments : GeoDataFrame
        The catchment data
    geomorpho_vars : list
        List of geomorphological variables
    landuse_vars : list
        List of land use variables
    output_dir : str
        Directory to save outputs
    """
    # Check if stream_order column exists
    stream_order_col = None
    if 'stream_order' in catchments.columns:
        stream_order_col = 'stream_order'
    elif 'ORDER' in catchments.columns:
        stream_order_col = 'ORDER'

    if stream_order_col is None:
        print("Stream order column (stream_order or ORDER) not found, skipping stream order analysis")
        return

    print(f"\nPerforming analysis by stream order using column '{stream_order_col}'...")

    # Create stream order categories
    catchments['order_category'] = pd.cut(
        catchments[stream_order_col],
        bins=[-float('inf'), 1, 2, float('inf')],
        labels=['Order 1', 'Order 2', 'Order 3+']
    )

    # Print stream order distribution
    order_counts = catchments['order_category'].value_counts()
    print("Stream order distribution:")
    print(order_counts)

    # Plot distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='order_category', data=catchments, order=['Order 1', 'Order 2', 'Order 3+'])
    plt.title('Distribution of Stream Orders')
    plt.xlabel('Stream Order')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'stream_order_distribution.png'), dpi=300)
    print(f"Saved stream order distribution to '{os.path.join(output_dir, 'stream_order_distribution.png')}'")

    # Compare key river characteristics by stream order
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(x='order_category', y='Google_Sin', data=catchments,
                order=['Order 1', 'Order 2', 'Order 3+'])
    plt.title('Current Sinuosity by Stream Order')
    plt.xlabel('Stream Order')
    plt.ylabel('Sinuosity')

    plt.subplot(1, 3, 2)
    sns.boxplot(x='order_category', y='sin_change_5920', data=catchments,
                order=['Order 1', 'Order 2', 'Order 3+'])
    plt.title('Sinuosity Change (1959-2020) by Stream Order')
    plt.xlabel('Stream Order')
    plt.ylabel('Sinuosity Change')

    plt.subplot(1, 3, 3)
    cat_counts = catchments.groupby(['order_category', 'sin_change_cat_5920']).size().unstack()
    cat_props = cat_counts.div(cat_counts.sum(axis=1), axis=0)
    cat_props.plot(kind='bar', stacked=True)
    plt.title('Sinuosity Change Categories by Stream Order')
    plt.xlabel('Stream Order')
    plt.ylabel('Proportion')
    plt.legend(title='Change Category')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'river_characteristics_by_stream_order.png'), dpi=300)
    print(
        f"Saved river characteristics by stream order to '{os.path.join(output_dir, 'river_characteristics_by_stream_order.png')}'")

    # Analyze factors influencing sinuosity for each stream order
    print("\nAnalyzing factors influencing sinuosity by stream order...")

    # Create a figure to plot the 6 most influential factors for each stream order
    plt.figure(figsize=(18, 15))

    # Create separate models for each stream order category
    order_categories = ['Order 1', 'Order 2', 'Order 3+']

    # Define results table to store model accuracies
    model_results = []

    for i, order_cat in enumerate(order_categories):
        # Subset data for this stream order
        subset = catchments[catchments['order_category'] == order_cat]

        # Skip if there are too few samples
        if len(subset) < 10:
            print(f"Skipping {order_cat} - insufficient samples ({len(subset)})")
            continue

        print(f"\nAnalyzing {order_cat} ({len(subset)} catchments)...")

        # Prepare data for modeling
        X_vars = geomorpho_vars.copy() + landuse_vars.copy()

        # Remove categorical variables
        X_vars = [var for var in X_vars if var in subset.columns and subset[var].dtype.kind in 'fcib']

        X = subset[X_vars].fillna(subset[X_vars].mean())
        y = subset['Google_Sin'].fillna(subset['Google_Sin'].mean())

        # Skip if there's not enough variation in the target
        if y.std() < 0.01:
            print(f"Skipping {order_cat} - insufficient variation in sinuosity")
            continue

        # Model evaluation metrics
        print("Evaluating Random Forest and Linear Regression models")

        # For Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # Cross-validation for Random Forest
        rf_cv_scores = cross_val_score(rf, X, y, cv=min(5, len(subset)), scoring='r2')
        rf_cv_rmse = np.sqrt(-cross_val_score(rf, X, y, cv=min(5, len(subset)), scoring='neg_mean_squared_error'))

        print(f"Random Forest CV R²: {rf_cv_scores.mean():.4f} (±{rf_cv_scores.std():.4f})")
        print(f"Random Forest CV RMSE: {rf_cv_rmse.mean():.4f} (±{rf_cv_rmse.std():.4f})")

        # For Linear Regression
        lr = LinearRegression()

        # Cross-validation for Linear Regression
        lr_cv_scores = cross_val_score(lr, X, y, cv=min(5, len(subset)), scoring='r2')
        lr_cv_rmse = np.sqrt(-cross_val_score(lr, X, y, cv=min(5, len(subset)), scoring='neg_mean_squared_error'))

        print(f"Linear Regression CV R²: {lr_cv_scores.mean():.4f} (±{lr_cv_scores.std():.4f})")
        print(f"Linear Regression CV RMSE: {lr_cv_rmse.mean():.4f} (±{lr_cv_rmse.std():.4f})")

        # Store results
        model_results.append({
            'Stream Order': order_cat,
            'RF R²': rf_cv_scores.mean(),
            'RF R² Std': rf_cv_scores.std(),
            'RF RMSE': rf_cv_rmse.mean(),
            'LR R²': lr_cv_scores.mean(),
            'LR R² Std': lr_cv_scores.std(),
            'LR RMSE': lr_cv_rmse.mean(),
            'Samples': len(subset)
        })

        # Train Random Forest model on full dataset
        rf.fit(X, y)
        rf_r2 = r2_score(y, rf.predict(X))
        rf_rmse = np.sqrt(mean_squared_error(y, rf.predict(X)))
        print(f"Random Forest (full data) - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")

        # Get feature importance
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Plot the 6 most important features for this stream order
        plt.subplot(3, 1, i + 1)
        sns.barplot(x='Importance', y='Feature', data=importance.head(6))
        plt.title(f'Top 6 Factors Influencing Sinuosity - {order_cat} (RF R²: {rf_cv_scores.mean():.2f})')

        # Also print the top factors
        print(f"Top factors for {order_cat}:")
        print(importance.head(6))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_factors_by_stream_order.png'), dpi=300)
    print(
        f"Saved influential factors by stream order to '{os.path.join(output_dir, 'top_factors_by_stream_order.png')}'")

    # Analysis for delta appearance
    print("\nAnalyzing factors influencing delta appearance by stream order...")

    # Create a figure for delta appearance factors
    plt.figure(figsize=(18, 15))

    # Results table for classification models
    delta_model_results = []

    for i, order_cat in enumerate(order_categories):
        # Subset data for this stream order
        subset = catchments[catchments['order_category'] == order_cat]

        # Skip if there are too few samples
        if len(subset) < 10:
            print(f"Skipping {order_cat} - insufficient samples ({len(subset)})")
            continue

        # Prepare data for modeling
        X_vars = geomorpho_vars.copy() + landuse_vars.copy()

        # Remove categorical variables
        X_vars = [var for var in X_vars if var in subset.columns and subset[var].dtype.kind in 'fcib']

        X = subset[X_vars].fillna(subset[X_vars].mean())
        y = subset['delta_appear_7420'].fillna(0)

        # Skip if there's not enough positive cases
        if y.sum() < 3:
            print(f"Skipping {order_cat} - insufficient delta appearances ({y.sum()})")
            continue

        # Model evaluation metrics
        print(f"Evaluating Random Forest Classifier for delta appearance - {order_cat}")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        # For Random Forest Classifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        # Cross-validation for Random Forest - use stratified CV if possible
        try:
            cv = min(5, len(subset))
            rf_cv_accuracy = cross_val_score(rf_clf, X, y, cv=cv, scoring='accuracy')
            rf_cv_auc = cross_val_score(rf_clf, X, y, cv=cv, scoring='roc_auc')
            rf_cv_f1 = cross_val_score(rf_clf, X, y, cv=cv, scoring='f1')

            print(f"Random Forest CV Accuracy: {rf_cv_accuracy.mean():.4f} (±{rf_cv_accuracy.std():.4f})")
            print(f"Random Forest CV AUC: {rf_cv_auc.mean():.4f} (±{rf_cv_auc.std():.4f})")
            print(f"Random Forest CV F1: {rf_cv_f1.mean():.4f} (±{rf_cv_f1.std():.4f})")

            # Store results
            delta_model_results.append({
                'Stream Order': order_cat,
                'RF Accuracy': rf_cv_accuracy.mean(),
                'RF AUC': rf_cv_auc.mean(),
                'RF F1': rf_cv_f1.mean(),
                'Positive Cases': y.sum(),
                'Total Samples': len(subset)
            })

            # Train Random Forest model on full dataset
            rf_clf.fit(X, y)
            rf_accuracy = accuracy_score(y, rf_clf.predict(X))
            print(f"Random Forest (full data) - Accuracy: {rf_accuracy:.4f}")

            # Get feature importance
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_clf.feature_importances_
            }).sort_values('Importance', ascending=False)

            # Plot the 6 most important features for this stream order
            plt.subplot(3, 1, i + 1)
            sns.barplot(x='Importance', y='Feature', data=importance.head(6))
            plt.title(f'Top 6 Factors Influencing Delta Appearance - {order_cat} (RF AUC: {rf_cv_auc.mean():.2f})')

            # Also print the top factors
            print(f"Top delta appearance factors for {order_cat}:")
            print(importance.head(6))

        except Exception as e:
            print(f"Error evaluating model for {order_cat}: {e}")
            continue

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_factors_by_stream_order.png'), dpi=300)
    print(f"Saved delta factors by stream order to '{os.path.join(output_dir, 'delta_factors_by_stream_order.png')}'")

    # Print summary tables
    print("\nSummary of Sinuosity Models by Stream Order:")
    results_df = pd.DataFrame(model_results)
    if not results_df.empty:
        print(results_df[['Stream Order', 'Samples', 'RF R²', 'RF RMSE', 'LR R²', 'LR RMSE']])

        # Save to CSV
        results_df.to_csv(os.path.join(output_dir, 'stream_order_sinuosity_models.csv'), index=False)
        print(f"Saved sinuosity model results to '{os.path.join(output_dir, 'stream_order_sinuosity_models.csv')}'")

    print("\nSummary of Delta Appearance Models by Stream Order:")
    delta_results_df = pd.DataFrame(delta_model_results)
    if not delta_results_df.empty:
        print(delta_results_df[['Stream Order', 'Positive Cases', 'Total Samples', 'RF Accuracy', 'RF AUC', 'RF F1']])

        # Save to CSV
        delta_results_df.to_csv(os.path.join(output_dir, 'stream_order_delta_models.csv'), index=False)
        print(f"Saved delta model results to '{os.path.join(output_dir, 'stream_order_delta_models.csv')}'")


def plot_top_influential_factors(catchments, geomorpho_vars, landuse_vars, output_dir):
    """
    Plot the 6 most influential factors for sinuosity, delta appearance, and dam appearance

    Parameters:
    -----------
    catchments : GeoDataFrame
        The catchment data
    geomorpho_vars : list
        List of geomorphological variables
    landuse_vars : list
        List of land use variables
    output_dir : str
        Directory to save outputs
    """
    print("\nCreating combined visualization of top influential factors...")

    # Prepare data for modeling
    X_vars = geomorpho_vars.copy() + landuse_vars.copy()

    # Remove categorical variables
    X_vars = [var for var in X_vars if var in catchments.columns and catchments[var].dtype.kind in 'fcib']

    X = catchments[X_vars].fillna(catchments[X_vars].mean())

    # Create a figure with 3 subplots for the 3 outcomes
    plt.figure(figsize=(18, 15))

    # 1. Current sinuosity
    y_sin = catchments['Google_Sin'].fillna(catchments['Google_Sin'].mean())
    rf_sin = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_sin.fit(X, y_sin)

    importance_sin = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_sin.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.subplot(3, 1, 1)
    sns.barplot(x='Importance', y='Feature', data=importance_sin.head(6))
    plt.title('Top 6 Factors Influencing Current Sinuosity')

    # 2. Delta appearance
    y_delta = catchments['delta_appear_7420'].fillna(0)
    rf_delta = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_delta.fit(X, y_delta)

    importance_delta = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_delta.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.subplot(3, 1, 2)
    sns.barplot(x='Importance', y='Feature', data=importance_delta.head(6))
    plt.title('Top 6 Factors Influencing Delta Appearance (1974-2020)')

    # 3. Dam appearance
    y_dam = catchments['dam_appear_7420'].fillna(0)
    rf_dam = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_dam.fit(X, y_dam)

    importance_dam = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_dam.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.subplot(3, 1, 3)
    sns.barplot(x='Importance', y='Feature', data=importance_dam.head(6))
    plt.title('Top 6 Factors Influencing Dam Appearance (1974-2020)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_influential_factors_combined.png'), dpi=300)
    print(f"Saved combined influential factors to '{os.path.join(output_dir, 'top_influential_factors_combined.png')}'")

    # Print the top factors for each outcome
    print("\nTop 6 factors influencing current sinuosity:")
    print(importance_sin.head(6))

    print("\nTop 6 factors influencing delta appearance:")
    print(importance_delta.head(6))

    print("\nTop 6 factors influencing dam appearance:")
    print(importance_dam.head(6))

    return importance_sin, importance_delta, importance_dam

# ====================================
# MAIN SCRIPT
# ====================================


# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
else:
    print(f"Using existing output directory: {output_dir}")

print("===== LACTOSE DELBECQ Catchment Analysis - Updated Version =====")
print(f"Analysis date: 2025-07-08 14:49:50")
print(f"User: adille")
print("Starting analysis script...")

# ====================================
# STEP 1-2: LOAD AND EXAMINE DATA
# ====================================

# Load the data
print("\nStep 1: Loading catchment data...")
catchments = gpd.read_file(data_path)
print(f"Loaded {len(catchments)} catchment records from the dataset")

# Examine data structure
print("\nStep 2: Examining data structure...")
print("Dataset columns:")
for col in catchments.columns:
    print(f"  - {col} ({catchments[col].dtype})")

# ====================================
# STEP 3: CHECK FOR MISSING VALUES
# ====================================

print("\nStep 3: Checking for missing values...")
missing = catchments.isna().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print("Columns with missing values:")
    print(missing_cols)
else:
    print("No missing values found in the dataset")

# ====================================
# STEP 4: BASIC STATISTICS
# ====================================

print("\nStep 4: Calculating basic statistics...")
numeric_cols = catchments.select_dtypes(include=[np.number]).columns
print(f"Summary statistics for {len(numeric_cols)} numeric columns")
print(catchments[numeric_cols].describe().T[['count', 'mean', 'min', 'max']])

# ====================================
# STEP 5: RENAME COLUMNS
# ====================================

print("\nStep 5: Renaming columns to improve usability...")
# First replace slashes with underscores
catchments.columns = [col.replace('/', '_') for col in catchments.columns]
# Then replace percent signs with "pct" to avoid formula issues
catchments.columns = [col.replace('%', 'pct') for col in catchments.columns]
print("Columns renamed successfully")

# Display updated column names
print("Updated column names:")
for i, col in enumerate(catchments.columns):
    print(f"  - {col}")

# ====================================
# STEP 6: CREATE TEMPORAL CHANGE VARIABLES
# ====================================

print("\nStep 6: Creating temporal change variables...")
# Calculate changes in sinuosity over time
catchments['sin_change_5974'] = catchments['73_74_Sin_'].astype(float) - catchments['57_59_Sin_'].astype(float)
catchments['sin_change_7420'] = catchments['Google_Sin'].astype(float) - catchments['73_74_Sin_'].astype(float)
catchments['sin_change_5920'] = catchments['Google_Sin'].astype(float) - catchments['57_59_Sin_'].astype(float)
print("Created sinuosity change variables for three time periods")

# Create categorical sinuosity change variables
catchments['sin_change_cat_5974'] = pd.cut(
    catchments['sin_change_5974'],
    bins=[-float('inf'), -0.2, 0.2, float('inf')],
    labels=['Decreased', 'Stable', 'Increased']
)
catchments['sin_change_cat_7420'] = pd.cut(
    catchments['sin_change_7420'],
    bins=[-float('inf'), -0.2, 0.2, float('inf')],
    labels=['Decreased', 'Stable', 'Increased']
)
catchments['sin_change_cat_5920'] = pd.cut(
    catchments['sin_change_5920'],
    bins=[-float('inf'), -0.2, 0.2, float('inf')],
    labels=['Decreased', 'Stable', 'Increased']
)
print("Created categorical sinuosity change variables for three time periods")

# Create binary indicators for appearance/disappearance of deltas and dams
catchments['delta_appear_5974'] = ((catchments['57_59_Delt'] == 0) & (catchments['73_74_Delt'] == 1)).astype(int)
catchments['delta_appear_7420'] = ((catchments['73_74_Delt'] == 0) & (catchments['Google_Del'] == 1)).astype(int)
catchments['delta_appear_5920'] = ((catchments['57_59_Delt'] == 0) & (catchments['Google_Del'] == 1)).astype(int)
catchments['dam_appear_5974'] = ((catchments['57_59_Bar_'] == 0) & (catchments['73_74_Bar_'] == 1)).astype(int)
catchments['dam_appear_7420'] = ((catchments['73_74_Bar_'] == 0) & (catchments['Google_Bar'] == 1)).astype(int)
catchments['dam_appear_5920'] = ((catchments['57_59_Bar_'] == 0) & (catchments['Google_Bar'] == 1)).astype(int)
print("Created binary indicators for delta and dam appearance")

# ====================================
# STEP 7: ANALYZE CORRELATIONS
# ====================================

print("\nStep 7: Analyzing correlations between variables...")

# Define variable groups
geomorpho_vars = ['alt_mean', 'alt_max', 'HI', 'slope_mean', 'slope_max', 'aspect_med',
                 'TWI_mean','TWI_max', 'SPI_mean', 'SPI_max', 'SPI_min', 'rugo_mean',
                  'rugo_min','rugo_max', 'nbr_knickp', 'KSN_mean','KSN_median','KSN_max',
                  'den_shallo','den_profon', 'litho_majo', 'susceptibi']
print(f"Geomorphological variables: {', '.join(geomorpho_vars)}")

# Update variable names to use the new column names (with pct instead of %)
landuse_vars = ['pct_bati', 'pct_foret', 'pct_vegetati', 'pct_cult', 'def_2014_pct']
# Add forest cover variables if they exist
for var in ['forest_59_pct', 'forest_74_pct', 'forest_20_pct',
            'forest_change_59_74', 'forest_change_74_20', 'forest_change_59_20']:
    if var in catchments.columns:
        landuse_vars.append(var)

print(f"Land use variables: {', '.join(landuse_vars)}")

river_vars = ['Google_Sin', '73_74_Sin_', '57_59_Sin_',
             'sin_change_5974', 'sin_change_7420', 'sin_change_5920']
print(f"River variables: {', '.join(river_vars)}")

# Correlation between geomorphology and river characteristics
print("\nCalculating correlation between geomorphology and river characteristics...")
# Handle categorical variables if needed
numeric_geomorpho = catchments[geomorpho_vars].select_dtypes(include=[np.number]).columns.tolist()
corr_geomorph = catchments[numeric_geomorpho + river_vars].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_geomorph.loc[numeric_geomorpho, river_vars], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation: Geomorphology vs River Characteristics')
plt.tight_layout()
plt.savefig(save_figure('geomorph_river_correlation.png'), dpi=300)
print(f"Saved correlation heatmap to '{save_figure('geomorph_river_correlation.png')}'")

# Correlation between land use and river characteristics
print("\nCalculating correlation between land use and river characteristics...")
corr_landuse = catchments[landuse_vars + river_vars].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_landuse.loc[landuse_vars, river_vars], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation: Land Use vs River Characteristics')
plt.tight_layout()
plt.savefig(save_figure('landuse_river_correlation.png'), dpi=300)
print(f"Saved correlation heatmap to '{save_figure('landuse_river_correlation.png')}'")

# ====================================
# STEP 8: MODEL COMPARISON FOR CONTINUOUS OUTCOMES
# ====================================

print("\nStep 8: Comparing Linear Regression and Random Forest for sinuosity...")

# Prepare data for modeling
X_data = catchments[geomorpho_vars + landuse_vars].copy()

# Handle missing values
X_data = X_data.fillna(X_data.mean())

# Handle categorical variables (litho_majo)
if X_data['litho_majo'].dtype == 'object':
    print("Converting categorical lithology to dummy variables...")
    X_data = pd.get_dummies(X_data, columns=['litho_majo'], drop_first=True)

# Current sinuosity modeling
y_current = catchments['Google_Sin'].fillna(catchments['Google_Sin'].mean())
compare_rf_linear(X_data, y_current, X_data.columns.tolist(), 'current_sinuosity', output_dir)

# 1959-2020 sinuosity change modeling
y_change = catchments['sin_change_5920'].fillna(catchments['sin_change_5920'].mean())
compare_rf_linear(X_data, y_change, X_data.columns.tolist(), 'sinuosity_change_5920', output_dir)

# ====================================
# STEP 9: CATEGORICAL SINUOSITY CHANGE ANALYSIS
# ====================================

print("\nStep 9: Analyzing categorical sinuosity changes...")

# Counts of different change categories
for period in ['5974', '7420', '5920']:
    cat_col = f'sin_change_cat_{period}'
    counts = catchments[cat_col].value_counts()
    print(f"\nSinuosity change categories for {period}:")
    print(counts)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    sns.countplot(x=cat_col, data=catchments, order=['Decreased', 'Stable', 'Increased'])
    plt.title(f'Sinuosity Change Categories ({period})')
    plt.ylabel('Number of Catchments')
    plt.xlabel('Change Category')
    plt.savefig(save_figure(f'sinuosity_change_categories_{period}.png'), dpi=300)
    print(f"Saved category counts to '{save_figure(f'sinuosity_change_categories_{period}.png')}'")

    # Random Forest for predicting change category
    print(f"\nRunning Random Forest classification for sinuosity change category {period}...")

    # Skip if there's only one category
    if len(counts) > 1:
        # Prepare data
        y_cat = catchments[cat_col].dropna()
        X_cat = X_data.loc[y_cat.index]

        # Train Random Forest classifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_cat, y_cat)

        # Feature importance
        feature_imp = pd.DataFrame({
            'Feature': X_cat.columns,
            'Importance': rf_clf.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
        plt.title(f'Top 15 Factors Influencing Sinuosity Change Category ({period})')
        plt.tight_layout()
        plt.savefig(save_figure(f'sinuosity_cat_importance_{period}.png'), dpi=300)
        print(f"Saved feature importance to '{save_figure(f'sinuosity_cat_importance_{period}.png')}'")

        # Cross-validation
        cv_scores = cross_val_score(rf_clf, X_cat, y_cat, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Confusion matrix on training set
        y_pred = rf_clf.predict(X_cat)
        cm = confusion_matrix(y_cat, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=rf_clf.classes_, yticklabels=rf_clf.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Sinuosity Change ({period})')
        plt.savefig(save_figure(f'sinuosity_cat_confusion_{period}.png'), dpi=300)
        print(f"Saved confusion matrix to '{save_figure(f'sinuosity_cat_confusion_{period}.png')}'")

        # Spatial distribution of sinuosity change categories
        plt.figure(figsize=(15, 10))
        catchments.plot(column=cat_col, categorical=True, legend=True, cmap='RdYlGn')
        plt.title(f'Spatial Distribution of Sinuosity Changes ({period})')
        plt.savefig(save_figure(f'spatial_sinuosity_cat_{period}.png'), dpi=300)
        print(f"Saved spatial map to '{save_figure(f'spatial_sinuosity_cat_{period}.png')}'")
    else:
        print(f"Skipping Random Forest for {period} - insufficient variation in categories")

# ====================================
# STEP 10: TEMPORAL TREND ANALYSIS
# ====================================

print("\nStep 10: Performing temporal trend analysis...")
sin_data, delta_data, dam_data = analyze_temporal_trends(catchments, output_dir)

# ====================================
# STEP 11: CLUSTER ANALYSIS
# ====================================

print("\nStep 11: Performing catchment clustering analysis...")
catchments, cluster_stats = cluster_catchments(catchments, geomorpho_vars, landuse_vars, output_dir)

# ====================================
# STEP 12: STREAM ORDER ANALYSIS
# ====================================

print("\nStep 12: Performing analysis by stream order...")
analyze_by_stream_order(catchments, geomorpho_vars, landuse_vars, output_dir)

# ====================================
# STEP 13: TOP INFLUENTIAL FACTORS
# ====================================

print("\nStep 13: Analyzing top influential factors...")
importance_sin, importance_delta, importance_dam = plot_top_influential_factors(catchments, geomorpho_vars, landuse_vars, output_dir)

# ====================================
# FINALIZATION
# ====================================

print("\n===== Analysis Complete =====")
print(f"Total catchments analyzed: {len(catchments)}")
print(f"All results have been saved as PNG files in: {output_dir}")

# Optional: Save the enhanced catchment data with new variables and clusters
enhanced_output = os.path.join(output_dir, 'enhanced_catchments.gpkg')
try:
    catchments.to_file(enhanced_output, driver='GPKG')
    print(f"Enhanced catchment data saved to: {enhanced_output}")
except Exception as e:
    print(f"Could not save enhanced catchment data: {e}")