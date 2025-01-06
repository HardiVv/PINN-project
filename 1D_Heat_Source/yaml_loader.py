import yaml
from general import FCN  # Assuming your FCN class is in a file named fcn.py

def load_config(file_path):
    """Load the YAML configuration file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config