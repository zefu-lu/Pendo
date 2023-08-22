from pathlib import Path

WORKSPACE_PATH = Path.home() / ".pendo"
CONFIG_PATH = WORKSPACE_PATH / "config.yaml"
CHROMA_PATH = WORKSPACE_PATH / "chroma"
TIMESTAMPS_PATH = WORKSPACE_PATH / "timestamps"

def initialize_workspace_paths():
    if not WORKSPACE_PATH.exists():
        WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
    if not CHROMA_PATH.exists():
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    if not TIMESTAMPS_PATH.exists():
        TIMESTAMPS_PATH.mkdir(parents=True, exist_ok=True)