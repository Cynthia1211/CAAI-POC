"""Classify example questions and print the selected intent for each one."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from intent_router import classify_intent


EXAMPLES = [
    "Who usually takes me to my clinic follow-up?",
    "What appointments do I have on my calendar tomorrow?",
    "What are common symptoms of seasonal allergies?",
    "please add tmr dinner at 6PM in my calendar",
]


def main() -> None:
    print("=" * 72)
    for query in EXAMPLES:
        classification = classify_intent(query)
        print(f"Question: {query}")
        print(f"LLM result: {classification.model_dump()}")
        print("-" * 72)
    print("=" * 72)


if __name__ == "__main__":
    main()
