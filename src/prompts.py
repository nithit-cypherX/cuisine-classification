"""
prompts.py
----------
Defines all LLM prompt templates used in the experiment.

Each function takes recipe text (and optionally retrieved examples)
and returns a fully formatted prompt string ready to send to the API.

Strategies:
  1. zero_shot        — No examples, label list only
  2. dynamic_few_shot — k retrieved examples from ChromaDB as context

DO NOT modify prompts mid-experiment.
If you want to test a new prompt, add a new function — never overwrite existing ones.
"""

from config import TARGET_CUISINES

# Human-readable label for display
CUISINE_LABELS = ", ".join([c.replace("_", " ").title() for c in TARGET_CUISINES])
# → Italian, Mexican, Indian, Thai, Japanese, Southern Us

# Normalised label list used for parsing outputs
LABEL_LIST = TARGET_CUISINES


def zero_shot(text: str) -> str:
    """
    Zero-shot prompt: no examples, just the label list and the recipe.

    Args:
        text : Ingredient string (e.g. "garlic, tomato, basil, olive oil")

    Returns:
        Formatted prompt string
    """
    return f"""Classify the following recipe into exactly one cuisine category.
Choose only from: {CUISINE_LABELS}.
Reply with the cuisine name only — no explanation.

Recipe ingredients: {text}
Cuisine:"""


def dynamic_few_shot(text: str, retrieved_examples: list[dict]) -> str:
    """
    Dynamic few-shot prompt: uses k examples retrieved from ChromaDB.
    Examples are different for every test query.

    Args:
        text               : Test recipe ingredient string
        retrieved_examples : List of dicts with keys 'text' and 'cuisine'
                             (returned by vectorstore.retrieve_similar)

    Returns:
        Formatted prompt string
    """
    examples_block = ""
    for ex in retrieved_examples:
        label = ex["cuisine"].replace("_", " ").title()
        examples_block += f'Ingredients: {ex["text"]}\nCuisine: {label}\n\n'

    return f"""Classify the following recipe into exactly one cuisine category.
Choose only from: {CUISINE_LABELS}.
Reply with the cuisine name only — no explanation.

Here are some examples:

{examples_block.strip()}

Now classify this recipe:
Ingredients: {text}
Cuisine:"""
