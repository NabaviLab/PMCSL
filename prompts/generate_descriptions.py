from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prompts.templates import CLASS_DESCRIPTIONS, GRADE_PROMPT_TEMPLATE


def generate_with_gemini(model_name, api_key):
    """Generate one fixed pathology description per grade."""
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError("Install google-genai.") from exc

    client = genai.Client(api_key=api_key)
    outputs = {}
    for grade, definition in CLASS_DESCRIPTIONS.items():
        prompt = GRADE_PROMPT_TEMPLATE.format(
            grade_name=f"{grade}: {definition}"
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        if not response.text:
            raise RuntimeError(f"Empty Gemini response for {grade}")
        outputs[grade] = response.text.strip()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gemini-2.5-pro")
    parser.add_argument(
        "--output",
        default="data/prompts/gleason_descriptions.json",
    )
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Set {args.api_key_env} before running.")

    outputs = generate_with_gemini(args.model_name, api_key)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Saved descriptions to {path}")


if __name__ == "__main__":
    main()
