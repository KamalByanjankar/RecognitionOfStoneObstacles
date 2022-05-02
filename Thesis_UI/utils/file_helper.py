
from glob import glob
import os
import json
from os.path import basename

import sys, os
from pathlib import Path

executable = '.'
if getattr(sys, 'frozen', False):
    # we are running in a bundle
    frozen = 'ever so'
    bundle_dir = sys._MEIPASS
    executable = sys.executable
    p = Path(executable)
    executable = str(p.parent)

def files_from_directory(directory):
    paths = glob(f'{directory}/*/*/*/*.csv')
    if not len(paths):
        paths = glob(f'{directory}/*/*/*.csv')
        if not len(paths):
            paths = glob(f'{directory}/*/*.csv')
    new_files_set = []
    for item in paths:
        item = item.replace('\\','/')
        new_files_set.append({
            'folder_name': int(item.split('/')[-2]),
            'file_name': basename(item),
            'absolute_path': item,
            'type': item.split('/')[-4].upper(),
            'model': item.split('/')[-3],
        })
    return new_files_set

def create_necessary_directories():
    config = get_config()
    data_directory = f"{executable}/{config['data']}"
    echo_directory = f"{executable}/{config['echo']}"
    feature_directory = f"{executable}/{config['feature']}"
    model_directory = f"{executable}/{config['model']}"
    cnn_directory = f"{executable}/{config['cnn']}"
    mlp_directory = f"{executable}/{config['mlp']}"

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    if not os.path.exists(echo_directory):
        os.mkdir(echo_directory)
    if not os.path.exists(feature_directory):
        os.mkdir(feature_directory)
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)
    if not os.path.exists(cnn_directory):
        os.mkdir(cnn_directory)
    if not os.path.exists(mlp_directory):
        os.mkdir(mlp_directory)   

def get_config():
    config_file = f'{executable}/config.json'
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    return config_data

def check_or_create_folder(config, freq):
    train_folder = f"{executable}/{config['echo']}/{freq}/train"
    test_folder = f"{executable}/{config['echo']}/{freq}/test"
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    train_folder = f"{executable}/{config['feature']}/{freq}/train"
    test_folder = f"{executable}/{config['feature']}/{freq}/test"
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    