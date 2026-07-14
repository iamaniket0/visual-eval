"""Image editing evaluation pipeline."""
from pathlib import Path

from src.core.utils import ROOT, load_yaml

CONFIG_DIR = ROOT / "config" / "edit"
OUTPUTS_DIR = ROOT / "outputs" / "edit"
PROMPTS_DIR = ROOT / "prompts" / "edit"


def load_models_config() -> dict:
    return load_yaml(CONFIG_DIR / "models.yaml")


def load_settings() -> dict:
    return load_yaml(CONFIG_DIR / "settings.yaml")
