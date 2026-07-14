"""Text-to-image evaluation pipeline."""

from src.core.utils import ROOT, load_yaml

CONFIG_DIR = ROOT / "config" / "t2i"
OUTPUTS_DIR = ROOT / "outputs" / "t2i"
PROMPTS_DIR = ROOT / "prompts" / "t2i"


def load_models_config() -> dict:
    return load_yaml(CONFIG_DIR / "models.yaml")


def load_settings() -> dict:
    return load_yaml(CONFIG_DIR / "settings.yaml")
