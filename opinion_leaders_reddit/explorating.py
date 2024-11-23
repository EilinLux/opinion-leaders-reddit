import plotly.express as px
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import norm
import traceback
from plotly import offline

import os 

class QuantitativeOpinionLeaderAnalyzer:
    """
    Analyzes Reddit data to identify and cluster opinion leaders.
    """

    def __init__(self, df, popularity_features, engagement_features, initiative_features, n_clusters= 6,  alpha = 0.000001):
        """
        Initializes the analyzer with the DataFrame and feature lists.
        """
        self.df = df
        self.popularity_features = popularity_features
        self.engagement_features = engagement_features
        self.initiative_features = initiative_features
        self.user_metrics = df.copy()
        self.alpha = alpha
        self.n_clusters = n_clusters
        

    def analyze_opinion_leaders(self, filename_prefix):
        """
        Performs the analysis, including PCA, clustering, and visualization.
        """
        # Standardize the data
        scaler = StandardScaler()
        self.user_metrics[self.popularity_features] = scaler.fit_transform(self.user_metrics[self.popularity_features])
        self.user_metrics[self.engagement_features] = scaler.fit_transform(self.user_metrics[self.engagement_features])
        self.user_metrics[self.initiative_features] = scaler.fit_transform(self.user_metrics[self.initiative_features])

        # Apply PCA with checks for empty components
        self.user_metrics = self._apply_pca(self.user_metrics, self.popularity_features, 'popularity_pc')
        self.user_metrics = self._apply_pca(self.user_metrics, self.engagement_features, 'engagement_pc')
        self.user_metrics = self._apply_pca(self.user_metrics, self.initiative_features, 'initiative_pc')

        # Generate scree plots
        # self._generate_scree_plots(self.user_metrics)

        # Apply clustering (k-means)
        kmeans = KMeans(n_clusters=self.n_clusters)  
        self.user_metrics['cluster'] = kmeans.fit_predict(self.user_metrics[['popularity_pc', 'engagement_pc', 'initiative_pc']])

        # Print cluster information
        self._print_cluster_info(filename_prefix)

        # Calculate distance from center
        self.user_metrics['distance_from_center'] = np.sqrt(
            self.user_metrics['popularity_pc']**2 +
            self.user_metrics['engagement_pc']**2 +
            self.user_metrics['initiative_pc']**2
        )


        # # Get the top 10 users farthest from the center
        # top_10_farthest = self.user_metrics.nlargest(10, 'distance_from_center')
        # self._create_scatter_plot(top_10_farthest)

        # Calculate z-scores and identify outliers
        self.user_metrics = self._calculate_zscores_and_outliers()
        self._create_scatter_plot_with_outliers(filename_prefix)

        return self.user_metrics

    def _apply_pca(self, df, features, column_name):
        """
        Applies PCA to the given features and handles cases with no components.
        """
        pca = PCA(n_components=0.95)
        transformed_data = pca.fit_transform(df[features])
        if transformed_data.shape[1] > 0:
            df[column_name] = transformed_data[:, 0]
        else:
            print(f"Warning: No principal components extracted for {column_name} features.")
            df[column_name] = 0
        return df

    # def _generate_scree_plots(self):
    #     """
    #     Generates scree plots for each PCA component.
    #     """
    #     pca_popularity = PCA(n_components=0.95)
    #     pca_popularity.fit_transform(self.user_metrics[self.popularity_features])
    #     pca_engagement = PCA(n_components=0.95)
    #     pca_engagement.fit_transform(self.user_metrics[self.engagement_features])
    #     pca_initiative = PCA(n_components=0.95)
    #     pca_initiative.fit_transform(self.user_metrics[self.initiative_features])

    #     plt.figure()
    #     plt.plot(np.cumsum(pca_popularity.explained_variance_ratio_))
    #     plt.xlabel('Number of Components')
    #     plt.ylabel('Cumulative Explained Variance')
    #     plt.title('Scree Plot for Popularity Features')
    #     plt.show()

    #     plt.figure()
    #     plt.plot(np.cumsum(pca_engagement.explained_variance_ratio_))
    #     plt.xlabel('Number of Components')
    #     plt.ylabel('Cumulative Explained Variance')
    #     plt.title('Scree Plot for Engagement Features')
    #     plt.show()

    #     plt.figure()
    #     plt.plot(np.cumsum(pca_initiative.explained_variance_ratio_))
    #     plt.xlabel('Number of Components')
    #     plt.ylabel('Cumulative Explained Variance')
    #     plt.title('Scree Plot for Initiative Features')
    #     plt.show()

    def _print_cluster_info(self,filename_prefix):
        """
        Prints information about the clusters.
        """
        print("###########################################")
        print(filename_prefix)
        print("###########################################")

        for cluster_id in range(3):
            cluster_data = self.user_metrics[self.user_metrics['cluster'] == cluster_id]
            print(f"Cluster {cluster_id}:")
            print(f"  Number of Redditors: {len(cluster_data)}")
            print(f"  Mean Popularity: {cluster_data['popularity_pc'].mean():.2f}")
            print(f"  Mean Engagement: {cluster_data['engagement_pc'].mean():.2f}")
            print(f"  Mean Initiative: {cluster_data['initiative_pc'].mean():.2f}")

    def _calculate_zscores_and_outliers(self):
            """
            Calculates z-scores and identifies outliers.
            """

            try:
                
                # Identify outliers based on p-value (e.g., p < 0.01)
                #  if there are more than one data points in the DataFrame        
                critical_z = np.abs(norm.ppf(self.alpha/2))  # Calculate critical z-value directly

                for metric in ['popularity_pc', 'engagement_pc', 'initiative_pc']:

                    # Calculate z-scores only if there is more than one data point
                    if len(self.user_metrics[metric]) > 1:  
                        self.user_metrics[f'{metric}_zscore'] = zscore(self.user_metrics[metric])

                    else:
                        # Handle the case where there are not enough data points for z-score calculation
                        self.user_metrics[f'{metric}_zscore'] = 0  # Or assign another appropriate value


                
                if len(self.user_metrics) > 1:
                  
                    # create a new column called is_outlier in the DataFrame (Boolean: T or F)
                    self.user_metrics['is_outlier'] = (
                        # compares the absolute z-score to the critical z-value. 
                        # If the absolute z-score is greater, it means the data point is an outlier.
                        (np.abs(self.user_metrics['popularity_pc_zscore']) > 
                            # critical z-value
                            critical_z # zscore(self.alpha/2)
                            
                        ) |
                        (np.abs(self.user_metrics['engagement_pc_zscore']) > critical_z) |
                        (np.abs(self.user_metrics['initiative_pc_zscore']) > critical_z)
                    )
                else:
                    # If there's only one data point in the DataFrame, it's not possible to identify outliers,
                    self.user_metrics['is_outlier'] = False
                    
            except:
                print(traceback.print_exc())

            return self.user_metrics
        

    def _create_scatter_plot(self, top_10_farthest):
        """
        Creates a 3D scatter plot of Redditors.
        """
        fig = px.scatter_3d(
            self.user_metrics,
            x='popularity_pc_zscore',
            y='engagement_pc_zscore',
            z='initiative_pc_zscore',
            color='cluster',
            title="3D Scatter Plot of Redditors by Popularity, Engagement, and Initiative",
            labels={
                'popularity_pc': 'Popularity (PC1)',
                'engagement_pc': 'Engagement (PC1)',
                'initiative_pc': 'Initiative (PC1)',
                'index': "Redditor"
            },
            hover_name=self.user_metrics.index,
            opacity=0.5,
        )
        fig.add_scatter3d(
            x=top_10_farthest['popularity_pc'],
            y=top_10_farthest['engagement_pc'],
            z=top_10_farthest['initiative_pc'],
            mode='markers',
            marker=dict(size=8, color='red'),
            hovertext=top_10_farthest.index,
        )
        fig.update_layout(
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
            scene=dict(aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=40),
            height=400, width=1000
        )
        fig.show()

    def _create_scatter_plot_with_outliers(self, filename_prefix):
            """
            Creates a 3D scatter plot highlighting outliers with a different color.
            """

            # Separate outlier data
            outliers = self.user_metrics[self.user_metrics['is_outlier']]
            regular_points = self.user_metrics[~self.user_metrics['is_outlier']]

            fig = px.scatter_3d(
                regular_points,  # Plot regular points first
                x='popularity_pc',
                y='engagement_pc',
                z='initiative_pc',
                color='cluster',
                title="3D Scatter Plot of Redditors by Popularity, Engagement, and Initiative",
                labels={
                    'popularity_pc': 'Popularity (PC1)',
                    'engagement_pc': 'Engagement (PC1)',
                    'initiative_pc': 'Initiative (PC1)',
                    'index': "Redditor"
                },
                hover_name=regular_points.index,
                hover_data={'popularity_pc_zscore': True,
                            'engagement_pc_zscore': True,
                            'initiative_pc_zscore': True,
                            'is_outlier': True},
                opacity=0.5,
            )

            # Add outliers with a different color (e.g., green)
            fig.add_scatter3d(
                x=outliers['popularity_pc'],
                y=outliers['engagement_pc'],
                z=outliers['initiative_pc'],
                mode='markers',
                marker=dict(size=8, color='red'),  # Highlight outliers in green
                hovertext=outliers.index,
                name='Outliers'
            )

            fig.update_layout(
                scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
                scene=dict(aspectmode='data'),
                margin=dict(l=0, r=0, b=0, t=40),
                height=400, width=1000
            )
            # fig.show('lifeExp.html')
            # html file
            self._check_folder_extistance(filename_prefix)
            offline.plot(fig, filename=f'{filename_prefix}/3d_scatterplot.html')

    def get_outliers_dataframe(self, filename_prefix):
        """
        Returns a DataFrame containing the outliers and their values.

        Returns:
            pandas.DataFrame: A DataFrame with the outliers and their 
                            'popularity_pc', 'engagement_pc', and 'initiative_pc' values.
        """

        # Check if 'is_outlier' column exists
        if 'is_outlier' not in self.user_metrics.columns:
            print("Error: 'is_outlier' column not found. Make sure to run outlier detection first.")
            return None

        # Extract outliers
        outliers_df = self.user_metrics[self.user_metrics['is_outlier']][
            ['popularity_pc', 'engagement_pc', 'initiative_pc']
        ].copy()  # Select relevant columns and create a copy
        outliers_df.to_csv(f"{filename_prefix}/outliers_df.csv")
        return outliers_df
    
    def save_users_metrics(self, filename_prefix):

        self._check_folder_extistance(filename_prefix)
        self.user_metrics.describe().to_csv(f"{filename_prefix}/user_metrics_test.csv")

    def _check_folder_extistance(self, filename_prefix):
        # Extract directory from filename_prefix
        directory = os.path.dirname(filename_prefix)

        # Create directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)