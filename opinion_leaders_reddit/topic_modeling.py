from gensim.corpora import Dictionary
from gensim.models import Phrases, LdaModel
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib
from pyvis.network import Network
import networkx as nx
from gensim.models import Phrases
import re 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from collections import defaultdict
import gensim

cmap = plt.colormaps['viridis']  

from bokeh.resources import INLINE
from bokeh.io import output_notebook

output_notebook(resources=INLINE) 

nltk.download('stopwords') 

import logging
import logging.config
import traceback

from opinion_leaders_reddit.configurating import config_regex_stopwords


# Load the configuration without the filename
logging.config.fileConfig('opinion_leader_config/logger.conf', disable_existing_loggers=False)





class TopicModeler:
    """
    A class for performing topic modeling on text data, including preprocessing,
    model training, and visualization.
    """

    def __init__(self, posts_data, language_data="it", text_preprocessing_type="lemmatization", 
                 no_below=5, no_above=0.1, keep_n=5000, additional_stopwords=None, comment_words_in=True, 
                config_regex_stopwords = config_regex_stopwords
                 ):
        """
        Initializes the TopicModeler with data and preprocessing parameters.

        Args:
            posts_data (pd.DataFrame): Dataset of post dictionaries containing 'title' and 'comment_bodies' keys.
            language_data (str, optional): Language code for preprocessing (default: "it").
            text_preprocessing_type (str, optional): Normalization method ("stemming", "lemmatization", or None) (default: "lemmatization").
            no_below (int, optional): Minimum document frequency for dictionary filtering (default: 5).
            no_above (float, optional): Maximum document frequency for dictionary filtering (default: 0.1).
            keep_n (int, optional): Maximum number of tokens to keep in the dictionary (default: 5000).
            additional_stopwords (list, optional): Additional stopwords to remove (default: None).
            comment_words_in (bool, optional): Whether to include comment text (default: True).
        """
        self.logger = logging.getLogger("reddit_topic_modeling_logger")

        self.posts_data = posts_data
        self.language_data = language_data
        self.text_preprocessing_type = text_preprocessing_type
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n
        self.config_regex_stopwords = config_regex_stopwords
        self.additional_stopwords = self.define_stopwords(additional_stopwords)
        self.comment_words_in = comment_words_in
        self.lda_model = None
        self.vis_topicmodeling_pyLDAvis  = None

        # Prepare the text data for LDA
        self.dictionary, self.corpus, self.all_texts = self.prepare_text_for_lda()
        

    def define_stopwords(self, additional_stopwords):
        
        if additional_stopwords is None:
            return additional_stopwords
        else: 
            if isinstance(additional_stopwords, list):
                additional_stopwords_list  = []
                for configuration in additional_stopwords:
                    try: 
                        additional_stopwords_list.extend(self.config_regex_stopwords[configuration])
                    except:
                        print(f"{configuration} key not in config_regex_stopwords: {self.config_regex_stopwords.keys()}")
                return additional_stopwords_list
            elif isinstance(additional_stopwords, str):
                try: 
                    return self.config_regex_stopwords[configuration]
                except:
                    print(f"{configuration} key not in config_regex_stopwords: {self.config_regex_stopwords.keys()}")
            else: 
                print("This type is not accepted")






    def preprocess_text(self, text, tokenize=True):
        """
        Preprocesses text for word embedding analysis (e.g., Word2Vec).

        This function performs the following steps:
        1. Low-case: all-low case

        2. Cleaning:
        * Removes URLs: Hyperlinks often contain irrelevant information.
        * Removes Punctuation: Punctuation usually doesn't carry semantic meaning in this context.
        * Removes Non-Alphanumeric Characters: This ensures that only words and numbers remain.
        * Removes Underscores for Hashtags: This splits hashtags into individual words.

        3. Tokenization: Splits the text into individual words.

        4. Stopword Removal: Filters out common English words like "the," "and," "in," which
        usually don't contribute much to the meaning of a sentence.

        5. Length Filtering: Keeps only words that are longer than 2 characters. This removes 
        very short words that might be less meaningful or noisy.

        6. Lowercasing: Converts all words to lowercase to make the analysis case-insensitive.

        7. Optional Stemming/Lemmatization: If enabled, this step reduces words to their base
        or root forms. Stemming is a faster but more aggressive approach, while lemmatization
        is more accurate but computationally expensive. You can choose either "stemming" or
        "lemmatization" by setting the `text_preprocessing_type` variable.

        Args:
            text: The input text string to be preprocessed.

        Returns:
            A list of preprocessed words.
        """

        language_mapping = {
        'da': 'danish',
        'de': 'german',
        'el': 'greek',
        'en': 'english',
        'es': 'spanish',
        'fi': 'finnish',
        'fr': 'french',
        'hu': 'hungarian',
        'it': 'italian',
        'nl': 'dutch',
        'pt': 'portuguese',
        'ro': 'romanian',
        'sl': 'slovene',
        'sv': 'swedish'
        }


        # Remove URLs and punctuation (including underscores for hashtags)
        text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+|[^A-Za-z0-9]+", " ", text)


        if language_mapping[self.language_data]:
            stop_words = set(stopwords.words(language_mapping[self.language_data]))
        else:
            stop_words = set()

        if self.additional_stopwords:
            low_case_additional_stopwords = {item.lower() for item in self.additional_stopwords}
            stop_words.update(low_case_additional_stopwords)
            
        if tokenize:
            #Tokenize
            words = word_tokenize(text)

            filtered_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]

            # Stemming o Lemming or None
            if self.text_preprocessing_type == "stemming" and language_mapping[self.language_data]:
                stemmer = SnowballStemmer(language_mapping[self.language_data])
                filtered_words = [stemmer.stem(word) for word in filtered_words]

            elif self.text_preprocessing_type == "lemmatization":
                lemmatizer = WordNetLemmatizer()
                filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]

            return filtered_words
        else:
            return text
    

    def prepate_text_for_nlp_modeling(self):
        """
        Combines and preprocesses text from a DataFrame, 
        including adding bigrams and trigrams for improved word representations.

        Args:
            df (pandas.DataFrame): DataFrame with 'post_title', 'post_text', and optionally 
                                'comments_bodies' (a list of comment bodies) columns.
            language_data: Language-specific data for preprocessing (e.g., stopwords).
            additional_stopwords (list, optional): A list of additional stopwords to remove (default: None).
            comment_words_in (bool, optional): If True, include words from comments in the analysis; 
                                            otherwise, use only titles (default: True).
            text_preprocessing_type (str, optional): use "stemming"; or use "lemmatization" (default: "lemmatization").

        Returns:
            list: A list of preprocessed sentences, including added bigrams and trigrams.
        """

        # Combine all text from posts and comments
        all_sentences = []
        for index, row in self.posts_data.iterrows():
            all_sentences.append(self.preprocess_text(row["post_title_and_text"]))

            if self.comment_words_in:
                for comment in row.get("comments_bodies", []):
                    all_sentences.append(self.preprocess_text(comment))

        # Add bigrams and trigrams (using gensim Phrases)
        bigram = Phrases(all_sentences, min_count=5, threshold=10)  # Find common bigrams
        trigram = Phrases(bigram[all_sentences], threshold=10)  # Find common trigrams
        all_sentences = list(trigram[bigram[all_sentences]])  # Apply the bigram and trigram models

        return all_sentences

    def prepare_text_for_lda(self):
        """
        Preprocesses text data for LDA topic modeling.

        Returns:
            tuple: A tuple containing the Gensim dictionary, corpus, and processed texts.
        """

        self.logger.info(f"Preprocessing text data for LDA topic modeling.")

        all_texts = self.prepate_text_for_nlp_modeling()
        dictionary = Dictionary(all_texts)
        dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=self.keep_n)
        corpus = [dictionary.doc2bow(text) for text in all_texts]
        return dictionary, corpus, all_texts
    
    def train_lda_model(self, num_topics=10, update_every=1, chunksize=100, passes=10, alpha= 'auto', per_word_topics = True ):
        """
        Trains an LDA model on the preprocessed data.

        Args:
            num_topics (int, optional): Number of topics to extract (default: 10).
            corpus: This is your collection of text documents converted into a bag-of-words format (a list of lists, where each inner list represents a document and contains tuples of (token\_id, token\_count)).
            id2word: A dictionary that maps token IDs to their corresponding words.
            num_topics: The number of topics you want the model to discover. This is often the most crucial parameter to tune.
            random_state: A seed for the random number generator, ensuring you get the same results each time you run the model with the same parameters and data.
            update_every: Controls how often the model parameters are updated. 1 means online learning (update after each document), while 0 means batch learning (update after the entire corpus).
            chunksize: The number of documents to be processed in each training batch.
            passes: The total number of passes through the entire corpus during training.
            alpha: The Dirichlet hyperparameter for the document-topic distribution. It controls the sparsity of the distribution (higher alpha means documents are more likely to contain a mix of many topics). 'auto' lets the model learn the optimal value.
            per_word_topics: If True, the model stores the topic distribution for each word in addition to the topic distribution for each document.
        Returns:
            gensim.models.LdaModel: The trained LDA model.
        """
        self.logger.info(f"Training an LDA model on the preprocessed data.")

        self.lda_model = LdaModel(
            corpus=self.corpus,  # The corpus of documents (list of list of (token_id, token_count) tuples)
            id2word=self.dictionary,  # Mapping from word IDs to words (dictionary)
            num_topics=num_topics,  # The number of topics to extract
            random_state=17,  # Random state for reproducibility
            update_every=update_every,  # Number of documents to be iterated through for each update. 1 = online learning, 0 = batch learning
            chunksize=chunksize,  # Number of documents to be used in each training chunk
            passes=passes,  # Total number of training passes through the corpus
            alpha=alpha,  # Dirichlet hyperparameter for document-topic distribution (controls document sparsity)
            per_word_topics=per_word_topics  # Whether to store the topic distribution for each word
        )

        return self.lda_model

    @staticmethod
    def generate_wordcloud_grid(lda_model, grid_dimensions="2x3"):
        """
        Generates and displays word clouds for each topic in a grid layout.

        Args:
            lda_model: Trained LDA model object (e.g., from Gensim)
            grid_dimensions (str): A string defining the grid layout in the format "rowsxcolumns" (e.g., "2x3" for a 2 rows by 3 columns grid).

        Raises:
            ValueError: If the grid dimensions are invalid for the number of topics.

        Returns:
            None
        """

        num_topics = lda_model.num_topics
        
        try:
            rows, cols = map(int, grid_dimensions.split("x"))
        except ValueError:
            raise ValueError("Invalid grid dimensions format. Use 'rowsxcolumns' (e.g., '2x3').")

        if rows * cols < num_topics:
            raise ValueError(f"Grid dimensions too small for {num_topics} topics. Please increase rows or columns.")

        # Get the total number of topics
        num_topics = lda_model.num_topics

        # Calculate the size of each sub-plot based on grid dimensions
        fig_width = 10 * cols  
        fig_height = 5 * rows  

        plt.figure(figsize=(fig_width, fig_height))

        for topic_idx, topic in enumerate(lda_model.print_topics(num_topics=num_topics, num_words=10)):  
            topic_words_freq = {}
            for word_weight_pair in topic[1].split("+"):
                weight, word = word_weight_pair.split("*")
                word = word.strip().replace('"', '')
                topic_words_freq[word] = float(weight)

            # Create the word cloud
            wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words_freq)

            # Plot the word cloud on the grid
            plt.subplot(rows, cols, topic_idx + 1)  # Create subplots according to the grid dimensions
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Topic #{topic_idx + 1}")

        plt.tight_layout()  # Adjust layout to prevent overlapping titles
        plt.show()


    def visualize_topics(self):
        # Visualize
        self.vis_topicmodeling_pyLDAvis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)
    
        return pyLDAvis.display(self.vis_topicmodeling_pyLDAvis)




    def plot_topic_subreddit_graph(self, filename_prefix, subreddit_list=None, num_top_words=10, num_subreddits_limit=None, 
                                        subreddits=None):
    
        """
        Creates and visualizes an interactive network graph of topics, top words, and subreddits.

        Args:posts_data
            subreddit_list (list): list of subreddits.
            num_top_words (int): Number of top words to display per topic (default: 10).
            num_subreddits_limit (int, optional): Limit the number of subreddits to include (default: None).
            subreddits (list, optional): List of specific subreddit names to include (default: None).
            folder (str): Folder to save the generated HTML file (default: "graphs").
            filename (str): Filename for the generated HTML file (default: "topic_subreddit_graph.html").

        Returns:
            str: Path to the generated HTML file.
        """
        self.logger.info(f"Plotting topics")
        filename=f"{filename_prefix}topic_subreddit_graph.html"

        if subreddit_list is None:
            subreddit_list = self.posts_data['subreddit'].unique().tolist()

        try:

            texts = []
            # Calculate topic assignments for each subreddit based on post content using the LDA model.
            topic_assignments = {}

            # Iterate over the DataFrame rows
            for index, row in self.posts_data.iterrows():
                texts = nltk.word_tokenize(row['post_title'].lower()) + \
                        nltk.word_tokenize(" ".join(row['comments_bodies']).lower())
                bow_vector = self.dictionary.doc2bow(texts)
                topics = self.lda_model.get_document_topics(bow_vector)

                subreddit = row["subreddit"]
                if subreddit not in topic_assignments:
                    topic_assignments[subreddit] = {}

                for topic_id, topic_prob in topics:
                    if topic_id not in topic_assignments[subreddit]:
                        topic_assignments[subreddit][topic_id] = 0
                    topic_assignments[subreddit][topic_id] += topic_prob

            # Filter subreddits if limits or specific list provided
            if num_subreddits_limit is not None:
                topic_assignments = dict(list(topic_assignments.items())[:int(num_subreddits_limit)])
            elif subreddits is not None:
                topic_assignments = {subreddit: topic_assignments[subreddit] for subreddit in subreddits if subreddit in topic_assignments}


            # Create the graph
            G = nx.Graph()

            # Add topic nodes
            for topic_idx in range(self.lda_model.num_topics):
                G.add_node(f"Topic {topic_idx + 1}", color="skyblue")

            # Add word nodes for each topic
            for topic_idx, topic_words in self.lda_model.print_topics(num_topics=self.lda_model.num_topics, num_words=num_top_words):
                for word_weight_pair in topic_words.split("+"):
                    weight, word = word_weight_pair.split("*")
                    word = word.strip().replace('"', '')
                    G.add_node(word, color="white")
                    G.add_edge(f"Topic {topic_idx + 1}", word)

            # Add subreddit nodes
            for subreddit_name in subreddit_list:
                G.add_node(subreddit_name, color="green")

            # Add edges between topics and subreddits
            for subreddit_name, topic_assignments in topic_assignments.items():
                for topic_idx, weight in topic_assignments.items():
                    G.add_edge(subreddit_name, f"Topic {topic_idx + 1}", weight=weight)


            # Create the Pyvis network with larger spacing
            net = Network(notebook=True, height="750px", width="100%", 
                        bgcolor="white", font_color="black", layout=True)

            net.force_atlas_2based(gravity=-10, central_gravity=0.01, spring_length=200, 
                                spring_strength=0.08, damping=0.4, overlap=1)

            net.from_nx(G)

            neighbor_map = net.get_adj_list()

            # Customize node colors with a colormap
            colormap = matplotlib.cm.get_cmap('tab20')  # Choose your preferred colormap
            topic_colors = [colormap(i) for i in np.linspace(0, 1, self.lda_model.num_topics)]
            subreddit_colors = [colormap(i) for i in np.linspace(0.5, 1, len(subreddit_list))]

            for node in net.nodes:
                if node['label'].startswith("Topic"):
                    topic_idx = int(node['label'].split()[1]) - 1
                    node['color'] = matplotlib.colors.to_hex(topic_colors[topic_idx])
                    node['size'] = 10 + 20 * sum(topic_assignments.get(node['label'], {}).values())
                    neighbors = neighbor_map[node["id"]]
                    # avoid to have the other topics connected
                    # node['title'] += " Reddits:<br>" + "<br>".join([item for item in neighbors if "Topic" not in item])


                elif node['label'] in subreddit_list:
                    subreddit_idx = subreddit_list.index(node['label'])
                    node['color'] = matplotlib.colors.to_hex(subreddit_colors[subreddit_idx])
                    node["shape"] = "box"
                    neighbors = neighbor_map[node["id"]]
                    # node["title"] += " Topics:<br>" + "<br>".join([item for item in neighbors if "Topic" in item])

                else:
                    node['color'] = 'gray' 


            net.show_buttons(filter_=['physics'])


            net.show(filename)
            # return net

        except Exception as e:
            print(f"An error occurred during graph visualization: {e}")
            print(traceback.format_exc())
            

    def save_model_and_data(self, filename_prefix, params):
        """
        Saves the trained LDA model, dictionary, corpus, and all_texts.
        Creates the necessary directories if they don't exist.
        Adds information from the params dictionary to the filenames.

        Args:
            filename_prefix (str): The prefix to use for the filenames.
            params (dict): A dictionary containing the parameters used for training the LDA model.
        """

        # Extract directory from filename_prefix
        directory = os.path.dirname(filename_prefix)

        # Create directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Construct filenames with parameter information
        def format_params(params):
            param_str = "_".join(f"{key}-{value}" for key, value in params.items())
            return param_str.replace(" ", "_").replace("'", "").replace('"', '')  # Remove spaces, quotes

        train_params = format_params(params['train_lda_model'])
        modeller_params = format_params(params['modeller'])

        model_filename = f"{filename_prefix}lda_model_{train_params}_{modeller_params}.model"
        dict_filename = f"{filename_prefix}dictionary_{train_params}_{modeller_params}.dict"
        corpus_filename = f"{filename_prefix}corpus_{train_params}_{modeller_params}.json"
        texts_filename = f"{filename_prefix}all_texts_{train_params}_{modeller_params}.txt"
        lda_pyLDAvis_filename =  f"{filename_prefix}lda_pyLDAvis_{train_params}_{modeller_params}.html"

        pyLDAvis.save_html(self.vis_topicmodeling_pyLDAvis, lda_pyLDAvis_filename)

        # Save the LDA model
        self.lda_model.save(model_filename)

        # Save the dictionary
        self.dictionary.save(dict_filename)

        # Save the corpus
        with open(corpus_filename, 'w') as f:
            json.dump(self.corpus, f)

        # Save all_texts
        with open(texts_filename, 'w', encoding='utf-8') as f:
            for text in self.all_texts:
                f.write(f"{text}\n")




class AuthorInfluenceAnalyzer:
    """
    Analyzes author influence within topics identified by an LDA model.

    Args:
        lda_model: Trained LDA model.
        dictionary: Gensim dictionary used for the LDA model.
        df: DataFrame with 'post_author', 'comment_author', 'post_title_and_text', 
            and 'comments_bodies' columns.
        num_top_words (int, optional): Number of top words to consider per topic (default: 10).
    """
    def __init__(self, lda_model, dictionary, df, num_top_words=10, saving_path = None):
        self.lda_model, self.dictionary  = self.load_model_and_data(lda_model, dictionary)
        self.df = df
        self.num_top_words = num_top_words
        self.topic_word_distributions = self.get_topic_word_distributions()
        self.save_path = self.extract_save_path(saving_path, lda_model)

    def extract_save_path(self,saving_path,  lda_model):
        if saving_path is None:
            if isinstance(lda_model, str):
                return f"{self.extract_folder_path(lda_model)}dominant_authors_graph.html"
            else:
                return "data/graphs/dominant_authors_graph.html"
        else:
            return saving_path

    @staticmethod
    def extract_folder_path(filepath):
        """
        Extracts the folder path from a given filepath using regular expressions.

        Args:
            filepath (str): The filepath to extract the folder path from.

        Returns:
            str: The extracted folder path.
        """

        # Regular expression to match the folder path
        pattern = r"^(.*/)"  # Matches everything up to the last slash

        match = re.match(pattern, filepath)
        if match:
            return match.group(1)  # Return the matched folder path
        else:
            return None  # Return None if no match is found



    def load_model_and_data(self, lda_model, dictionary):
        """
        Loads the trained LDA model, dictionary, corpus, and all_texts from files.

        Args:
            filename_prefix (str): The prefix used for the filenames.
        """

        if isinstance(lda_model, gensim.models.LdaModel) and isinstance(dictionary, gensim.corpora.Dictionary):
            return lda_model, dictionary
        elif isinstance(lda_model, str) and isinstance(dictionary, str):
            # self.logger.info(f"Loading trained LDA model, dictionary, corpus, and all_texts.")

            print("I am here")
            try:
                # Load the LDA model and  the dictionary
  
                return LdaModel.load(lda_model), Dictionary.load(dictionary)

                # # Load the corpus
                # with open(f"{filename_prefix}_corpus.json", 'r') as f:
                #     self.corpus = json.load(f)

                # # Load all_texts
                # with open(f"{filename_prefix}_all_texts.txt", 'r', encoding='utf-8') as f:
                #     self.all_texts = [line.strip() for line in f]


                # self.logger.info(f"Successfully loaded LDA model and data with prefix '{filename_prefix}'")

            except FileNotFoundError:
                print(f"Error: One or more files with prefix '{filename_prefix}' not found.")
            except Exception as e:
                print(f"Error loading LDA model and data: {e}")


    def get_topic_word_distributions(self):
        """
        Extracts the word distributions for each topic from the LDA model.

        Returns:
            list: A list of dictionaries, where each dictionary represents a topic and 
                  contains word-probability pairs.
        """
        topic_word_distributions = []
        for topic_idx in range(self.lda_model.num_topics):
            topic_words = self.lda_model.show_topic(topic_idx, topn=self.num_top_words)
            topic_word_dist = {word: prob for word, prob in topic_words}
            topic_word_distributions.append(topic_word_dist)
        return topic_word_distributions

    def analyze_author_influence(self, top_percentage):
        """
        Identifies influential authors for each topic based on post/comment proportions, 
        average topic probabilities, and keyword usage.

        Returns:
            dict: A dictionary where keys are topic indices and values are lists of 
                  influential authors for that topic.
        """
        author_topic_contributions = defaultdict(lambda: defaultdict(lambda: {'posts': 0, 'comments': 0, 'topic_probs': []}))
        author_keyword_counts = defaultdict(lambda: defaultdict(int))

        for _, row in self.df.iterrows():
            post_author = row['post_author']
            post_text = row['post_title_and_text']
            comments = row['comments_bodies']
            comment_authors = row['comments_author']  
            subreddit = row['subreddit']

            # Analyze post
            self.process_text(post_text, post_author, author_topic_contributions, author_keyword_counts, subreddit)

            # Analyze comments, iterating through both lists simultaneously
            for comment, comment_author in zip(comments, comment_authors):
                self.process_text(comment, comment_author, author_topic_contributions, author_keyword_counts, subreddit)


        # Identify dominant authors for each topic
        influential_authors_per_topic = {}
        for topic_idx in range(self.lda_model.num_topics):
            influential_authors = self.find_dominant_authors(topic_idx, author_topic_contributions, author_keyword_counts, top_percentage)
            influential_authors_per_topic[topic_idx] = influential_authors

        return influential_authors_per_topic

    def process_text(self, text, author, author_topic_contributions, author_keyword_counts, subreddit):
        """
        Processes a single text (post or comment) to update author contributions and keyword counts.
        """
        bow_vector = self.dictionary.doc2bow(text.lower().split())
        topic_probs = self.lda_model.get_document_topics(bow_vector)

        for topic_idx, prob in topic_probs:

            author_topic_contributions[author][topic_idx].setdefault('subreddits', set()).add(subreddit)

            author_topic_contributions[author][topic_idx]['topic_probs'].append(prob)
            if 'post' in text:
                author_topic_contributions[author][topic_idx]['posts'] += 1
            elif 'comment' in text:
                author_topic_contributions[author][topic_idx]['comments'] += 1

            # Count keywords
            for word, _ in self.topic_word_distributions[topic_idx].items():
                if word in text.lower():
                    author_keyword_counts[author][topic_idx] += 1


    def find_dominant_authors(self, topic_idx, author_topic_contributions, author_keyword_counts, top_percentage = 0.1 ):
            """
            Determines dominant authors for a specific topic and returns their scores.
            """
            print(author_topic_contributions)
            author_scores = {}
            for author, topic_data in author_topic_contributions.items():
                if topic_idx in topic_data:
                    topic_prob = np.mean(topic_data[topic_idx]['topic_probs'])
                    post_count = topic_data[topic_idx]['posts']
                    comment_count = topic_data[topic_idx]['comments']
                    keyword_count = author_keyword_counts[author][topic_idx]

                    # # Topic initiation (check if author was the first to mention the topic)
                    # topic_initiation_score = 0
                    # if topic_data[topic_idx].get('first_mention', False):
                    #     topic_initiation_score = 0.2  # Adjust weight as needed

                    # # Topic spread (check if author discussed the topic in multiple subreddits)
                    # topic_spread_score = 0
                    # if len(topic_data[topic_idx].get('subreddits', [])) > 1:
                    #     print("here")
                    #     topic_spread_score = 10  # Adjust weight as needed


                    # TODO: Calculate a combined score (adjust weights as needed)
                    """ use also the fact if they have initiate the topic, if they have spread this in on subreddit or multiple"""
                    score = (
                        0.4 * topic_prob + 
                        0.3 * (post_count + comment_count) + 
                        0.3 * keyword_count 
                        # + topic_spread_score + 
                        # topic_initiation_score
                    )
                    author_scores[author] = score

            sorted_authors = sorted(author_scores.items(), key=lambda item: item[1], reverse=True)
            num_influencers = max(1, int(top_percentage * len(sorted_authors)))  #  at least 1 influencer

            # Return a list of tuples (author, score)
            return [(author, score) for author, score in sorted_authors[:num_influencers]]
    
    def visualize_dominant_authors(self, influential_authors_per_topic, exclude_authors = []):
        """
        Visualizes dominant authors and their associated topics in a network graph.


        """
        G = nx.Graph()

        for topic_idx in influential_authors_per_topic:
            scores = [score for _, score in influential_authors_per_topic[topic_idx]]
            
            # Calculate outliers using IQR (you can use other methods)
            Q1 = np.percentile(scores, 25)
            Q3 = np.percentile(scores, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            for tuple_author_score in influential_authors_per_topic[topic_idx]:
                author, score = tuple_author_score[0], tuple_author_score[1]
                # Exclude authors
                if author in exclude_authors:
                    continue


                if topic_idx:
                    topic_label = f"Topic {topic_idx + 1}\n" + ", ".join([f"{word}:{self.topic_word_distributions[topic_idx][word]:.2f}" 
                                                                        for word in self.topic_word_distributions[topic_idx]
                                                                        ]) 
                    G.add_node(f"Topic_{topic_idx}", label=topic_label)

                    author_node = f"Author_{author}"



                    G.add_node(author_node, label=f"{author}\n(Score: {score:.2f})")  # Add size attribute
                    G.add_edge(f"Topic_{topic_idx}", author_node, weight=score)



        # Create and customize the Pyvis network
        net = Network(notebook=True, height="750px", width="100%", bgcolor="white", font_color="black")
        net.from_nx(G)

        # Adjust node size based on score (optional)
        for node in net.nodes:
            if node['id'].startswith("Author_"):
                score = float(node['label'].split("(Score: ")[1].rstrip(")"))
                node['size'] = 10 + score
                if score < lower_bound or score > upper_bound:
                    node['font'] = {'size': 5 + score, 'color': "black", 'strokeColor': "white", 'strokeWidth': 2} 
                    node["shape"] = "star"
                    node["color"] = "#ec7215"  # Outlier color
                else:
                    node["color"] = "#1B9E77"   # Not Outlier color
      

            elif node['id'].startswith("Topic_"):
                node["shape"] = "box"               
                node["color"] = "#c6c2f6"  # Set topic color (purplish)
                node['font'] = {'size': 20, 'color': "black", 'strokeColor': "white", 'strokeWidth': 2} 


        net.show_buttons(filter_=['physics'])
        
        net.show(self.save_path) 

        return net