import pandas as pd
import json 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt



class OpinionLeadershipProcessor:

    def __init__(self, df):
        self.df = df

    @staticmethod
    def _explode_cols(dataframe_partition,  cols = ['comments_author', 'comments_upvote_score', 'comments_bodies']):
        return dataframe_partition.explode(cols)
    
    
    @staticmethod
    def _add_is_self_comment_col(df_exploded):
        df_exploded['is_self_comment'] = df_exploded['post_author'] == df_exploded['comments_author']
        df_exploded['is_self_comment'] = df_exploded['is_self_comment'].astype(str) 
        return df_exploded
    


    def prepare_data_for_opinion_leadership_analysis(self, dataframe_partition):
        """
        Prepares the data for opinion leadership analysis by:
          - Exploding the DataFrame to have one row per comment, 
          - Adding a boolean column indicating if a comment is from the post author,
          - Filtering out deleted Redditors (authors and commentors).

        Args:
          dataframe_partition (pandas.DataFrame): The input DataFrame containing post and comment data.

        Returns:
          pandas.DataFrame: A DataFrame prepared for opinion leadership analysis.
        """

        # Explode the DataFrame to have one row per comment
        df_exploded = self._explode_cols(dataframe_partition) 

        # Create a new column to indicate if the comment author is the same as the post author
        df_exploded = self._add_is_self_comment_col(df_exploded)

#         df_exploded = df_exploded[~df_exploded['comments_author'].isin(["[deleted]",None])]
#         df_exploded = df_exploded[~df_exploded['post_author'].isin(["[deleted]",None])]
        
        return df_exploded

    def scale_columns(self, df_exploded, measures, typ="normalization"):
        
        subreddits_list = self.df['subreddit'].unique()
        
        for measure in measures:
            for subreddit in subreddits_list:

                # extract a subset based on the subreddit
                subreddit_data = df_exploded[df_exploded['subreddit'] == subreddit]

                if typ == "normalization":
                    # Calculate the range for the subreddit for the specific measure
                    subreddit_range = subreddit_data[measure].max() - subreddit_data[measure].min()

                    if subreddit_range != 0:  # Check if range is not zero
                        # Normalize the measures in the subreddit dataset using Min-Max scaling
                        df_exploded.loc[df_exploded['subreddit'] == subreddit, f"{measure}_normalized"] = (
                            subreddit_data[measure] - subreddit_data[measure].min()
                        ) / subreddit_range

                elif typ == "standardization":
                    # calculate the std for the subreddit for the specific measure
                    subreddit_std = subreddit_data[measure].std()

                    if subreddit_std != 0:  # Check if standard deviation is not zero
                        # standardize the measures in the subreddit dataset 
                        df_exploded.loc[df_exploded['subreddit'] == subreddit, f"{measure}_std"] = (
                            subreddit_data[measure] - subreddit_data[measure].mean()
                        ) / subreddit_std

        return df_exploded

    def calculate_user_metrics(self, df_exploded):

        # Calculate metrics, separating by author and it is their own or not (is_self_comment)
        user_metrics = df_exploded.groupby(['post_author', 'is_self_comment']).agg(

            # Engagement
            mean_comments_per_post=('comments_num_normalized', 'mean'),  
            std_comments_per_post=('comments_num_normalized', 'std'),  
            
            # Popularity
            mean_post_upvote_ratio=('post_upvote_ratio_normalized', 'mean'),  
            std_post_upvote_ratio=('post_upvote_ratio_normalized', 'std'),  
            mean_comment_upvote_score=('comments_upvote_score_normalized', 'mean'),  
            std_comment_upvote_score=('comments_upvote_score_normalized', 'std'),
            # mean_comment_upvote_ratio=('comments_upvote_ratio_normalized', 'mean'),  
            # std_comment_upvote_ratio=('comments_upvote_ratio_normalized', 'std'),

        ).unstack(fill_value=0)

        # Flatten the MultiIndex columns
        user_metrics.columns = ['_'.join(col).strip() for col in user_metrics.columns.values]

        col_list = user_metrics.columns
        replace_dict = {}
        for col_name in col_list:
            if "_False" in col_name:
                replace_dict[col_name] = col_name.replace("_False","_on_others_posts")
            if "_True" in col_name:
                replace_dict[col_name] = col_name.replace("_True","_on_own_posts")
            else:
                pass
        user_metrics = user_metrics.rename(columns=replace_dict)

        # Add other metrics that don't require separation by is_self_comment
        user_metrics = user_metrics.join(df_exploded.groupby('post_author').agg(
            # number of unique subreddits each author has posted
            subreddits_participation =('subreddit', 'nunique'),  # Initiation (proactiveness)
            
            new_threads = ('post_id', 'nunique'),  # Initiation (proactiveness)
        ))

        # Normalize the 'new_threads' column
        user_metrics['subreddits_participation_normalized'] = (user_metrics['subreddits_participation'] - user_metrics['subreddits_participation'].min()) / (user_metrics['subreddits_participation'].max() - user_metrics['subreddits_participation'].min())
        user_metrics['new_threads_normalized'] = (user_metrics['new_threads'] - user_metrics['new_threads'].min()) / (user_metrics['new_threads'].max() - user_metrics['new_threads'].min())

        user_metrics = user_metrics.astype(float).fillna(0) # Fill NaN values with 0
        
        return user_metrics



class OpinionLeaderDataPreparator:

    def __init__(self):
        """
        Initializes the OpinionLeaderIdentifier with the provided posts data.

        Args:
            posts_data (list): A list of dictionaries, where each dictionary represents a post and contains
                                 information about the author, subreddit, and comments.
        """
        self.posts_data = {}
        self.partitioned_data = {}
        self.df = None
    
    def head(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.df.head()

    @classmethod    
    def load_posts_data(self, filename):
        """
        
        filename="data/raw_data/20240712_101844_posts_data_full_posts_data" 

        """
        with open(filename, 'r') as f:
            posts_data = json.load(f)

            # Convert datetime strings back to datetime objects
            for post in posts_data:
                if not isinstance(post['post_created_utc'], datetime):  # Check if it's NOT a datetime
                    post['post_created_utc'] = datetime.fromisoformat(post['post_created_utc'])
        self.posts_data = posts_data
        self.df = pd.DataFrame(self.posts_data)


        

    @classmethod    
    def filter_and_partition_posts(self, election_date_strings = ['25-05-2014', '26-05-2019', '09-06-2024']):
        """
        Filters posts based on multiple 5-year windows around the given dates, partitions by date and language, 
        and returns a dictionary of DataFrames.

        Args:
            date_strings (list): A list of date strings in the format 'DD-MM-YYYY'.

        Returns:
            dict: A dictionary where keys are tuples (date_string, language) and values are DataFrames 
                  containing the filtered and partitioned posts.
        """
        partitioned_data = {}

        # Combined text and title for the post
        self.df = self.combine_post_text(self.df)

        for date_string in election_date_strings:
            try:
                target_date = datetime.strptime(date_string, '%d-%m-%Y')
            except ValueError:
                raise ValueError("Invalid date format. Use 'DD-MM-YYYY'.")

            start_date = target_date - timedelta(days=365 * 2.5)
            end_date = target_date + timedelta(days=365 * 2.5)

            # Filter the DataFrame directly
            filtered_df = self.df[(self.df['post_created_utc'] >= start_date) & (self.df['post_created_utc'] <= end_date)]




            # Partition by language
            for language in filtered_df['post_language'].unique():
                partitioned_data[(date_string, language)] = filtered_df[filtered_df['post_language'] == language]
        
        
        self.partitioned_data = partitioned_data
        # print keys 
        print(self.partitioned_data.keys())


    @staticmethod
    def combine_post_text(df):
        # Combine 'post_title' and 'post_text' into a single text column
        print(len(df.columns))
        df['combined_post_text'] = df['post_title'] + ' ' + df['post_text']
        print(len(df.columns))

        return df 

    @classmethod
    def get_dataset(self, election_date = '25-05-2014', language= 'en'):
        """_summary_

        Args:
            election_date (str, optional): _description_. Defaults to '25-05-2014'.
            language (str, optional): _description_. Defaults to 'en'.

        Returns:
            _type_: _description_
        """
        return self.partitioned_data[(election_date, language )]


    def prepare_data_for_opinion_leadership_analysis(self, dataframe_partition):
        """
        Prepares the data for opinion leadership analysis by:
          - Exploding the DataFrame to have one row per comment, 
          - Adding a boolean column indicating if a comment is from the post author,
          - Filtering out deleted Redditors (authors and commentors).

        Args:
          dataframe_partition (pandas.DataFrame): The input DataFrame containing post and comment data.

        Returns:
          pandas.DataFrame: A DataFrame prepared for opinion leadership analysis.
        """

        # Explode the DataFrame to have one row per comment
        df_exploded = self._explode_cols(dataframe_partition) 

        # Create a new column to indicate if the comment author is the same as the post author
        df_exploded = self._add_is_self_comment_col(df_exploded)

#         df_exploded = df_exploded[~df_exploded['comments_author'].isin(["[deleted]",None])]
#         df_exploded = df_exploded[~df_exploded['post_author'].isin(["[deleted]",None])]
        
        return df_exploded
    
