
import yaml

def load_config(config_file):
  """
  Loads a YAML configuration file.

  Args:
    config_file (str): Path to the YAML configuration file.

  Returns:
    dict: The loaded configuration as a dictionary.
  """
  with open(config_file, 'r') as f:
      config = yaml.safe_load(f)
  return config

# Load the configuration
app_config = load_config("opinion_leader_config/app_config.yaml")  
lda_config = load_config("opinion_leader_config/lda_modeling_config.yaml")  
config_regex_stopwords = load_config("opinion_leader_config/stop_words_config.yaml")