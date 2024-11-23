from datetime import datetime
import pandas as pd
import os
import praw 
import time
import langdetect 
import json 

from datetime import datetime
import os 
import logging
import logging.config

from collections import defaultdict, Counter


# Load the configuration without the filename
logging.config.fileConfig('opinion_leader_config/logger.conf', disable_existing_loggers=False)


class RedditPostsCollector():

    """
    RedditPostsCollector: Collects posts and comments from Reddit.
    """

    def __init__(self, reddit:praw.Reddit, id:str, subreddit_details_df: pd.DataFrame(), subreddit_counts={}, post_ids_seen=set(), ):
        self.reddit = reddit
        self.data_structure = [] 
        self.post_ids_seen = post_ids_seen
        self.subreddit_details_df = subreddit_details_df
        self.id = id
        self.subreddit_counts = subreddit_counts  

        self.logger = logging.getLogger("reddit_posts_collector_logger")


 
    def _parse_date(self, date_str):
        """
        Parses a date string in 'DD-MM-YYYY' format into a datetime object.

        Args:
            date_str (str): The date string to parse.

        Returns:
            datetime or None: A datetime object representing the parsed date, or None if parsing fails.
        """
        try:
            if date_str:
                return datetime.strptime(date_str, '%d-%m-%Y')
            return None
        except ValueError:
            self.logger.error(f"Invalid date format: {date_str}. Please use DD-MM-YYYY.")
            return None

    def _get_subreddits_to_analyze(self, num_subreddits_limit, exclude_subreddits_limit):
        """
        Determines the subreddits to analyze based on limits and exclusions.

        Args:
            num_subreddits_limit (int or None): Maximum number of subreddits to analyze.
            exclude_subreddits_limit (list or None): List of subreddits to exclude.

        Returns:
            dict: A dictionary containing the subreddits to analyze and their post counts.
        """
        # Get all subreddits if no limits are specified
        subreddits_to_analyze = self.subreddit_counts.copy() 

        if exclude_subreddits_limit:
            subreddits_to_analyze = {
                subreddit: count
                for subreddit, count in subreddits_to_analyze.items()
                if subreddit not in exclude_subreddits_limit
            }

        if num_subreddits_limit:
            subreddits_to_analyze = dict(list(subreddits_to_analyze.items())[:num_subreddits_limit])

        return subreddits_to_analyze



    def _is_valid_submission(self, submission, start_date, end_date, language):
        """
        Checks if a submission meets the specified date and language filters.

        Args:
            submission: The PRAW submission object.
            start_date (datetime or None): The start date for filtering.
            end_date (datetime or None): The end date for filtering.

        Returns:
            bool: True if the submission is valid, False otherwise.
        """
        if submission.id in self.post_ids_seen:
            self.logger.info(f"Skipping duplicate post ID: {submission.id}")
            return False 
        
        if start_date or end_date:
            post_date = datetime.utcfromtimestamp(submission.created_utc)

            if start_date and post_date < start_date:
                return False
            if end_date and post_date > end_date:
                return False
        
        try:
            if langdetect.detect(submission.title) != language:
                return False
        except langdetect.lang_detect_exception.LangDetectException:
            print(f'[Warn] Language detection failed for post ID: [{submission.id}] {submission.title}')
            return False

        return True

    def _classify_type(self, submission):
        # Classify post type based on URL
        if submission.is_self:
            post_type = 'text'
        elif any(submission.url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            post_type = 'image'
        elif submission.url.endswith('.gifv'):  # Special case for gifv
            post_type = 'gifv'
        elif 'v.redd.it' in submission.url or submission.url.endswith('.mp4'):
            post_type = 'video'
        elif submission.url.startswith('https://www.reddit.com/gallery/'):
            post_type = 'gallery'
        else:
            post_type = 'link'  # Default to link if no other type is identified
        return post_type

    def _extract_subreddit(self, submission):
        return submission.subreddit.display_name
    
    def _extract_id(self, submission):
        return submission.id
    
    def _extract_title(self, submission):
        return submission.title.encode('utf-8').decode('utf-8')
    
    def _extract_author(self, submission):
        
        return submission.author.name if submission.author else None  # Handle deleted authors
    
    def _extract_text(self, submission):
        return submission.selftext.encode('utf-8').decode('utf-8')
    
    def _combine_title_and_text(self, submission):
        return f"{submission.title.encode('utf-8').decode('utf-8')} {submission.selftext.encode('utf-8').decode('utf-8')}"
    
    def _extract_creation_date(self, submission):
        return datetime.utcfromtimestamp(submission.created_utc).isoformat()

    def _extract_upvote_score(self, submission):
        return submission.score
    
    def _extract_upvote_ratio(self, submission):
        return submission.upvote_ratio
    
    def _extract_post_data(self, submission, language, term):
        """
        Extracts relevant data from a Reddit post.

        Args:
            submission (praw.models.Submission): The PRAW submission object representing the Reddit post.

        Returns:
            dict: A dictionary containing the extracted post data.
        """
        try:


            post_data = {
                'subreddit': self._extract_subreddit(submission),
                'post_id': self._extract_id(submission),
                'post_title': self._extract_title(submission),
                'post_author': self._extract_author(submission),
                'post_text': self._extract_text(submission),
                'post_title_and_text': self._combine_title_and_text(submission),
                'post_created_utc': self._extract_creation_date(submission),
                'post_upvote_score': self._extract_upvote_score(submission),
                'post_upvote_ratio': self._extract_upvote_ratio(submission),
                'post_url': submission.url, 
                'post_language': language, 
                'search_term': term,  
                'post_type': self._classify_type(submission),
                'comments_bodies': [],
                'comments_lenght': [],  
                'comments_author': [],
                'comments_upvote_score': [],
                'comments_created_utc': [],
                'comments_num': 0,
            }
            return post_data
        
        except Exception as e:
            self.logger.error(f"Error extracting post data for post ID {submission.id}: {e}")
            return None
    
    def __extract_details(self, comment, post_data, comments_num):
        try:
            post_data['comments_bodies'].append(comment.body)
            post_data['comments_lenght'].append(len(comment.body))
            post_data['comments_created_utc'].append(datetime.utcfromtimestamp(comment.created_utc).isoformat() if comment.created_utc else None)
            post_data['comments_upvote_score'].append(comment.score)
            post_data['comments_author'].append(comment.author.name if comment.author else None)
            comments_num += 1 
            return post_data, comments_num
        except: 
            import traceback
            print(traceback.print_exc())
               
    def _extract_comments_data(self, submission, post_data, check_comments_language, language):
            """
            Extracts relevant data from comments of a Reddit post, including 
            the graph representation in the desired format.

            Args:
                submission: The PRAW submission object.
                post_data (dict): The dictionary containing the extracted post data.
                check_comments_language (bool): Whether to filter comments by language.
                language (str): The language to filter comments by.

            Returns:
                dict: The updated post_data dictionary with extracted comments data.
            """
            comments_num = 0
            comments_graph = {}  # Initialize an empty comment graph



            # Build comment graph recursively
            def build_comment_tree(comment, graph):
                comment_id = comment.id
                parent_id = comment.parent_id.split('_')[1] if comment.parent_id else None
                
                comment_data = {
                    'author': comment.author.name if comment.author else None,
                    'id': comment_id,
                    'body': comment.body,
                    'created_utc': datetime.utcfromtimestamp(comment.created_utc).isoformat() if comment.created_utc else None,
                    'children': [],  # Initialize an empty list for children
                    'parent_id': parent_id,  # Add parent_id label
                    'upvote_score':comment.score
                    }

                graph[comment_id] = comment_data  # Add the comment to the graph

                if parent_id:
                    if parent_id not in graph:
                        print("parent id:", parent_id)
                        # Create parent node for the post_id parent (since it doesn't exist)
                        graph[parent_id] = {'children': []} 
                        graph[parent_id]["root"] = True 

                    graph[parent_id]['children'].append(comment_id)  # Add comment ID to parent's children

                for reply in comment.replies:
                    build_comment_tree(reply, graph)  # Recursively build replies


            def _extract_comment_details(comment, post_data, comments_num):
                if check_comments_language:
                    try:
                        if langdetect.detect(comment.body) == language:
                            post_data, comments_num = self.__extract_details(comment, post_data, comments_num)
                    except langdetect.lang_detect_exception.LangDetectException:
                        self.logger.warning(f"Language detection failed for comment ID: [{comment.id}] {comment.body}")
                else:
                    post_data, comments_num = self.__extract_details(comment, post_data, comments_num)
                return post_data, comments_num
            

            
            # build post tree
            for top_level_comment in submission.comments:
                build_comment_tree(top_level_comment, comments_graph)

            # extract comments details for the post
            for comment in submission.comments.list():
                try:
                    post_data, comments_num = _extract_comment_details(comment, post_data, comments_num)
                except Exception as e:
                    self.logger.error(f"Error extracting comment data for comment ID {comment.id}: {e}")

            post_data["comments_num"] = comments_num
            post_data['comments_graph'] = comments_graph  # Add the graph to post_data
            
            return post_data


    
    def save_data(self,  filename, format="csv", folder="data/raw_data", data_structure=None):
        """
        Saves the data structure to a file.

        Args:
            format (str, optional): The desired file format. Defaults to "csv".
                Supported formats are "csv", "json", and "txt".
            filename (str, optional): The base name of the output file (without extension).
                Defaults to "posts_data".

        Raises:
            ValueError: If an invalid format is provided.
        """
        if not (data_structure):
            posts_data = self.data_structure
   
        else: 
            print("[Skip] Data structure is empy")
      
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Add a timestamp to the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{folder}/{timestamp}_{filename}"
        if format == "csv":

            if not isinstance(posts_data, pd.DataFrame):
                try: 
                    posts_data = pd.DataFrame(posts_data)
                except Exception as e:
                    print(f"[Error]: {e}")
            posts_data.to_csv(f"{filename}.csv", index=False)
        elif format == "json":

            if len(posts_data)>0:

                with open(filename, 'w') as f:
                    json.dump(posts_data, f)
            else:
                print("[Skip] Data structure is empy")

        else:
            raise ValueError("Invalid format. Choose from 'csv' or 'json'.")
        
    def load_file(format="csv", filename="posts_data", folder="data/raw_data"):
        """
        Loads post data from a file into a Pandas DataFrame.

        Args:
            format (str, optional): The file format to load. Defaults to "csv".
                Supported formats: "csv", "json", "pickle".
            filename (str, optional): The base filename (without extension). Defaults to "posts_data".
            folder (str, optional): The directory containing the file. Defaults to "data/raw_data".

        Returns:
            pandas.DataFrame: The loaded post data.

        Raises:
            ValueError: If an invalid format is specified.
        """
        filename = f"{folder}/{filename}"

        if format == "csv":
            return pd.read_csv(f"{filename}.csv")
        elif format == "json":
            return pd.read_json(f"{filename}.json", orient="records")
        elif format == "pickle":
            return pd.read_pickle(f"{filename}.pkl")
        else:
            raise ValueError("Invalid format. Choose from 'csv', 'json', or 'pickle'.")
        
    
    def save_posts_data(self, filename="posts_data", format="json", folder="data/raw_data"):  
        """
        Saves the collected post data to a file in the specified format.

        This method persists the accumulated post data stored in the `self.data_structure` 
        attribute to a file. It leverages the functionality of the parent class's `save_data` 
        method to handle file writing. The filename is constructed to include a timestamp
        and a unique instance ID to prevent overwriting and enhance traceability.

        Args:
            filename (str, optional): The base name of the output file (without extension).
                Defaults to "posts_data".
            format (str, optional): The desired file format ("csv" or "json"). Defaults to "json".
            folder (str, optional): The directory in which to save the file. Defaults to "data/raw_data".
    
        Returns:
            None
        """

        filename = f"posts_data_{self.id}_{filename}"
        self.save_data(format=format, folder=folder, filename=filename)        


    def extract_subreddit_posts_and_comments(
        self,
        config,                               # Configuration dictionary mapping languages to search terms
        num_subreddits_limit=None,            # Maximum number of subreddits to process (optional)
        exclude_subreddits_limit=None,        # List of subreddits to exclude (optional)
        subreddit_post_limit=None,               # Maximum number of posts to fetch per subreddit per search term (optional)
        subreddit_comments_limit=None,           # Maximum number of comments to fetch per post (optional)
        start_date=None,                      # Start date for post filtering (optional, format: 'DD-MM-YYYY')
        end_date=None,                        # End date for post filtering (optional, format: 'DD-MM-YYYY')
        check_comments_language=False,        # Whether to filter comments by language (optional)
        save_on_the_fly=True,                 # Save data for each subreddit after processing (optional)
        save_at_the_end=True,                 # Save all data at the end of processing (optional)
        reset_research=False                  # Reset previously seen post and subreddit IDs (optional)

    ):
        """
        Extracts and analyzes posts and comments from Reddit based on the provided configuration.

        Args:
            config: A dictionary mapping language codes (e.g., 'en', 'fr') to lists of search terms for each language.
            num_subreddits_limit: The maximum number of subreddits to process. Defaults to None (no limit).
            exclude_subreddits_limit: A list of subreddits to exclude from the analysis. Defaults to None.
            subreddit_post_limit: The maximum number of posts to fetch per subreddit per search term. Defaults to 1.
            subreddit_comments_limit: The maximum number of comments to fetch per post. Defaults to 3.
            start_date: The start date for post filtering (inclusive). Defaults to None (no start date filter).
            end_date: The end date for post filtering (inclusive). Defaults to None (no end date filter).
            check_comments_language: Whether to filter comments by the specified language. Defaults to False.
            save_on_the_fly: Whether to save data for each subreddit after processing. Defaults to True.
            save_at_the_end: Whether to save all data at the end of processing. Defaults to True.
            reset_research: Whether to reset the sets tracking seen post and subreddit IDs. Defaults to False.

        Returns:
            A tuple containing:
                - A list of dictionaries, each representing a single post and its comments.
                - A dictionary of subreddit names and the number of analyzed posts per subreddit.
        """

        private_subreddit_list = []

        start_date = self._parse_date(start_date)
        end_date = self._parse_date(end_date)

        self.logger.info("Analysis started")

        subreddits_to_analyze = self._get_subreddits_to_analyze(num_subreddits_limit, exclude_subreddits_limit)

        if reset_research:
            self.post_ids_seen = set()

        total_subreddits = len(subreddits_to_analyze)
        counter = 0 
        for subreddit_name in subreddits_to_analyze:
            counter = counter + 1

            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                self.logger.info(f'Analyzing "{subreddit_name}" subreddit ({counter}/{total_subreddits})')

                for language, terms in config.items():
                    for term in terms:
                        
                        self.logger.info(f'Searching "{term}" for "{language}" language')

                        # Use a try-except block to catch potential errors during search
                        try:
                            for submission in subreddit.search(term.encode('utf-8'), limit=subreddit_post_limit):
                                if not self._is_valid_submission(submission, start_date, end_date, language):
                                    continue

                                post_data = self._extract_post_data(submission, language, term)
                                if post_data is None:  # Skip if post data extraction failed
                                    continue

                                submission.comments.replace_more(limit=subreddit_comments_limit)
                                post_data = self._extract_comments_data(submission, post_data, check_comments_language, language)

                                self.data_structure.append(post_data)
                                self.subreddit_counts[subreddit_name] += 1
                                self.post_ids_seen.add(submission.id)
                        except Exception as e:
                            self.logger.error(f"Error searching for term '{term}' in subreddit '{subreddit_name}': {e}")
                            continue  # Continue to the next term


                if save_on_the_fly:
                    self.logger.info(f'Saving results for "{subreddit_name}"')
                    self.save_posts_data(filename=f"_{subreddit_name}")  # Use subreddit_name in filename

            except Exception as e:
                if str(e) == 'received 403 HTTP response':
                    self.logger.error(f"Access to subreddit '{subreddit_name}' forbidden: {e}")
                    private_subreddit_list.append(subreddit_name)
                    continue  # Skip this subreddit if it's private

                elif str(e) == 'received 429 HTTP response':
                    self.logger.warning(f"Rate limit reached for subreddit '{subreddit_name}': {e}")
                    private_subreddit_list.append(subreddit_name)
                    time.sleep(2)  # Wait for 2 seconds before retrying

                else:
                    self.logger.error(f"Error searching in {subreddit_name}: {e}")
                    # Consider adding a retry mechanism here

        # Delete private subreddits from data collector (if necessary)
        if private_subreddit_list:
            self.subreddit_datacollector.delete_subreddits(private_subreddit_list)


        self._add_subreddit_counter_columns()
        # # add the value counter to the df 
        # self.subreddit_details_df["counter"] = self.subreddit_details_df["subreddit"].map(self.subreddit_counts)


        # # # Calculate total counts per subreddit
        # # self.subreddit_counts = Counter(self.df['subreddit'])
        # self.subreddit_details_df["total_counter"] = self.subreddit_details_df["subreddit"].map(self.subreddit_counts)

        # # Calculate counts for each post type
        # post_type_counts = self.data_structure.groupby('subreddit')['post_type'].value_counts().unstack(fill_value=0)

        # # Add post type counts as columns to the DataFrame
        # for post_type in post_type_counts.columns:
        #     self.subreddit_details_df[post_type] = self.subreddit_details_df["subreddit"].map(post_type_counts[post_type])
        
        if save_at_the_end:
            self.save_posts_data()

        return self.data_structure, self.subreddit_details_df
    

    def _add_subreddit_counter_columns(self):
        """
        Adds a 'total_counter' column with the total count of posts per subreddit,
        and separate columns for each post type with their respective counts.
        """

        # Calculate total counts per subreddit
        self.subreddit_counts = Counter(post['subreddit'] for post in self.data_structure)
        self.subreddit_details_df["total_counter"] = self.subreddit_details_df["subreddit"].map(self.subreddit_counts)


class SubredditsDataCollector():
    def __init__(self, reddit: praw.Reddit):
        self.reddit = reddit  # Store the PRAW Reddit instance
        self.private_subreddit_list = []
        self.subreddit_details_df = pd.DataFrame()
        self.subreddit_counts = {}
        # Use the configured logger for this class
        self.logger = logging.getLogger("subreddits_data_collector_logger")


    def search_subreddits(self, config, excluded_subreddits=[], limit=None):
        """
        Searches for relevant subreddits in multiple languages and initializes a counter for each.

        Args:
            config (dict): A dictionary mapping language codes (e.g., 'en', 'fr') to lists of search terms for each language.
            excluded_subreddits (list): A list of subreddit names to exclude from the search.
            limit (int or None): The maximum number of subreddits to return per search term. Defaults to None (no limit).

        Returns:
            None
        """
        for language, term_list in config.items():
            for term in term_list:
                term = term.encode('utf-8')
                for subreddit in self.reddit.subreddits.search(term, limit=limit):
                    if subreddit.display_name.lower() not in excluded_subreddits:
                        self.subreddit_counts[subreddit.display_name] = 0
        self.logger.info(f"Selected subreddits: {len(self.subreddit_counts)}")


    def filter_subreddits(self, top_n=None, subreddits=None):
        """
        Filters a dictionary of subreddit counts based on either the top N subreddits or a specific list.

        Args:
            top_n (int, optional): The number of top subreddits to return (default: None).
            subreddits (list, optional): A list of subreddit names to include (default: None).

        Returns:
            dataframe: A dictionary containing the filtered subreddits and their counts.

        Raises:
            ValueError: If both top_n and specific_subreddits are provided or if neither is provided.

        """
        if top_n and subreddits:
            raise ValueError("[Error] Cannot specify both top_n and subreddits")
        if not top_n and not subreddits:
            raise ValueError("[Error] Must specify either top_n or subreddits")

        if top_n:
            self.subreddit_counts =  dict(
                sorted(
                    self.subreddit_counts.items(), key=lambda item: item[1], reverse=True
                )[:top_n]
            )
        else:
            self.subreddit_counts = {
                subreddit: count
                for subreddit, count in self.subreddit_counts.items()
                if subreddit in subreddits
            }

        if not self.subreddit_details_df.empty: 
            self.subreddit_details_df = self.subreddit_details_df[self.subreddit_details_df['subreddit'].isin(self.subreddit_counts.keys())]


        return self.subreddit_details_df 
    
    def get_subreddit_counts(self):
        return self.subreddit_counts

    def _create_timestamped_filename(self, folder, filename):
        """
        Creates a timestamped filename within a specified folder.

        Args:
            folder (str): The path to the folder where the file will be saved.
            filename (str): The base filename to which the timestamp will be added.

        Returns:
            str: The full path to the timestamped filename.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)  # Create the folder if it doesn't exist

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Generate timestamp
        filename = f"{folder}/{timestamp}_{filename}"  # Construct the full filename

        return filename
            
    def get_subreddit_details(self, folder="data/subreddits_details", filename = "subreddits_details"):
        """
        Fetches various information about subreddits from Reddit using the PRAW library.

        Args:
            reddit (praw.Reddit): An authenticated PRAW Reddit instance.
            subreddits (list): A list of subreddit names to fetch data for.

        Returns:
            pandas.DataFrame: A DataFrame containing the following columns:
                - subreddit: The name of the subreddit.
                - created_utc: The date and time the subreddit was created (in UTC).
                - subscribers: The number of subreddit subscribers.
                - description: The public description of the subreddit.
                - over18: Whether the subreddit is marked as Not Safe For Work (NSFW).

        Raises:
            praw.exceptions.RedditAPIException: If there's an error communicating with the Reddit API (e.g., invalid subreddit name).

        """
        
        all_data = []
        subreddits = self.subreddit_counts.keys()
        for subreddit_name in subreddits:
            self.logger.info(f"Fetching data for subreddit: {subreddit_name}")

            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                all_data.append({
                    "subreddit": subreddit.display_name,
                    "created_utc": datetime.utcfromtimestamp(subreddit.created_utc).date(),
                    "subscribers": subreddit.subscribers,
                    "description": subreddit.public_description or "",
                    "over18": subreddit.over18,
                })

            except Exception as e:
                if str(e) == "received 403 HTTP response":
                    self.logger.error(f"[Error] Access to subreddit '{subreddit_name}' forbidden: {e}")
                    self.private_subreddit_list.append(subreddit_name)
                    continue  # Skip this subreddit if it's private
            
                else:
                    self.logger.error(f"[Error] Fetching data for subreddit: {subreddit_name}, Error: {e}")
        
        self.delete_subreddits()
        
        self.subreddit_details_df = pd.DataFrame(all_data).sort_values(by=['subscribers'], ascending=False)

        # add the value counter to the df 
        #self.subreddit_details_df["counter"] = self.subreddit_details_df["subreddit"].map(self.subreddit_counts)

        print("Saving Data")
        filename = self._create_timestamped_filename(folder, filename)
        self.subreddit_details_df.to_csv(f"{filename}.csv", index=False)



    def delete_subreddits(self, private_subreddit_list = []):
        """
        Removes subreddits .
        """
        private_subreddit_list = private_subreddit_list + self.private_subreddit_list

        for key in private_subreddit_list:

            self.subreddit_counts.pop(key, None)
    
        try:
            # case SubredditsDataCollector
            if not self.subreddit_details_df.empty:
                self.subreddit_details_df = self.subreddit_details_df[~self.subreddit_details_df['subreddit'].isin(private_subreddit_list)] 
        except:
            # Other cases 
            pass

