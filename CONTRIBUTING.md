# Contributing to Visual Eval

## Quick Start

```bash
git clone git@github.com:your-org/visual-eval.git
cd visual-eval-
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your API keys
pytest                  # verify everything works
```

## Adding a New T2I Generator

1. Create `src/t2i/generators/your_model.py`:

```python
from src.core.registry import register

@register("your-model-id")
async def generate(prompt: str, output_path: str, **kwargs) -> str:
    """Generate an image from a text prompt.

    Args:
        prompt: The text prompt.
        output_path: Where to save the generated image.

    Returns:
        The path to the saved image.
    """
    # Call your model's API
    # Save the result to output_path
    return output_path
```

2. Add model config to `config/t2i/models.yaml`:

```yaml
your-model-id:
  display_name: "Your Model Name"
  provider: your_provider
  tier: sanity          # sanity | full | all
  cost_per_image: 0.04  # USD estimate
```

3. Register the import in `src/t2i/generators/__init__.py`.

4. Run the sanity check:

```bash
visual-eval t2i generate --models sanity --dry-run
```

## Adding a New Edit Model

Same pattern as T2I, but in `src/edit/editors/`:

```python
from src.core.registry import register

@register("your-editor-id")
async def edit(source_path: str, instruction: str, output_path: str, **kwargs) -> str:
    # Apply the edit
    return output_path
```

Add config to `config/edit/models.yaml` and register in `src/edit/editors/__init__.py`.

## Project Structure

```
src/
├── core/           # Shared: registry, cost tracker, judge, utils
├── t2i/            # Text-to-image: generators, prompts, aggregator, report
│   └── generators/ # One file per model, @register decorator
├── edit/           # Image editing: editors, aggregator, report
│   └── editors/    # One file per model, @register decorator
└── cli.py          # Typer CLI entry point
```

## Code Standards

- **Python 3.10+** — use `X | Y` union syntax, not `Optional[X]`
- **Absolute imports** — `from src.core.utils import ...`, not relative
- **Ruff** for formatting and linting: `ruff format . && ruff check .`
- **Type hints** on all public functions
- No comments unless the *why* is non-obvious

## Running Tests

```bash
pytest                          # all tests
pytest tests/test_judge.py -v   # specific file
pytest -k "test_cost"           # pattern match
```

## Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] Linter passes (`ruff check .`)
- [ ] Formatter passes (`ruff format --check .`)
- [ ] New model has config entry in the appropriate `models.yaml`
- [ ] `.env.example` updated if new API keys are needed
