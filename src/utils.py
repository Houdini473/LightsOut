import yaml
import pickle
import json
from pathlib import Path
import glob 
import os


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

def load_pickles(dir_path):
    """Load folder of pickle files into single variable"""
    file_paths = glob.glob(os.path.join(dir_path, "*"))
    loaded_data = []

    for file_path in file_paths:
        data = load_pickle(file_path)
        # print(f"type(data) = {type(data)}")
        data_list = data['data_list'] if isinstance(data, dict) else data.data_list
        # print(f"type(data_list) = {type(data_list)}")
        # print(f"pickle length loaded: {len(data_list)}")
        loaded_data.extend(data_list)

    return loaded_data


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
