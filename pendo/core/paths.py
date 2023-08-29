from pathlib import Path
import pkg_resources
import shutil
import logging
import os

WORKSPACE_PATH = Path.home() / ".pendo"
CONFIG_PATH = WORKSPACE_PATH / "config.yaml"
CHROMA_PATH = WORKSPACE_PATH / "chroma"
TIMESTAMPS_PATH = WORKSPACE_PATH / "timestamps"

CREDENTIALS_PATH = WORKSPACE_PATH / "credentials"
CREDENTIALS_GMAIL_PATH = CREDENTIALS_PATH / "gmail_credentials.json"

ALL_CREDENTIALS_PATHS = [
    CREDENTIALS_GMAIL_PATH
]

TOKENS_PATH = WORKSPACE_PATH / "tokens"

def initialize_workspace_paths():
    if not WORKSPACE_PATH.exists():
        WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
    if not CHROMA_PATH.exists():
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    if not TIMESTAMPS_PATH.exists():
        TIMESTAMPS_PATH.mkdir(parents=True, exist_ok=True)
    if not CREDENTIALS_PATH.exists():
        CREDENTIALS_PATH.mkdir(parents=True, exist_ok=True)
    for path in ALL_CREDENTIALS_PATHS:
        if not path.exists():
            try:
                source = pkg_resources.resource_filename(__name__, "/".join(str(path).split("/")[-2:]))
                shutil.copy(source, path)
            except Exception as e:
                logging.error(f'Failed to initialize credentials file: {e}')
                raise e
    
    if not TOKENS_PATH.exists():
        TOKENS_PATH.mkdir(parents=True, exist_ok=True)

