import yaml
import pickle
import json
from pathlib import Path


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_pickle(obj, filepath):
    """Save object as pickle"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj, filepath):
    """Save object as JSON"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
