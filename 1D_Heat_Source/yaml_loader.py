import yaml


def load_config(file_path):
    """Load the YAML configuration file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config
