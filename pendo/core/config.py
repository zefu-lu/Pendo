import os
import logging
import pkg_resources
import shutil
import yaml

from .paths import CONFIG_PATH
from typing import Dict
from pathlib import Path

def load_config(config_path: Path = CONFIG_PATH) -> Dict:
    if not os.path.exists(config_path):
        logging.info(f'Config file not found at {config_path}, initialize new config file at {config_path}')
        _initialize_config(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
            config = {}
    return config

def _initialize_config(config_path: Path = CONFIG_PATH):
    if not os.path.exists(os.path.dirname(config_path)):
        try:
            os.makedirs(os.path.dirname(config_path))
        except Exception as e:
            logging.error(f'Failed to create directory {os.path.dirname(config_path)}: {e}')
    
    try:
        source = pkg_resources.resource_filename(__name__, 'config.yaml')
        shutil.copy(source, config_path)
        logging.info(f'Config file initialized at {config_path}')
    except Exception as e:
        logging.error(f'Failed to initialize config file: {e}')
        raise e