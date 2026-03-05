"""
Demo script for the keyword extractor tool.

This script demonstrates how an AI agent (or any Python program) can use the
`keyword_extractor` tool defined in `tool.py`.

What this demo covers:
- **Successful execution** with multiple sample texts (different domains / lengths).
- **Error handling** for invalid inputs (empty text, invalid `top_n`, wrong types).
- **A simple “agent workflow”** that processes text, calls the tool, and returns a
  consistent JSON-friendly structure even when errors occur.

Run:
    python demo.py
"""

from tool import keyword_extractor


def run_successful_cases() -> None:
    """
    Run the keyword extractor on multiple valid test cases.

    Prints:
        The extracted keyword tokens and (for the first test case) the metadata that
        describes the extraction run.
    """
    print("=" * 60)
    print("SUCCESSFUL EXECUTION – Test cases")
    print("=" * 60)

    # Test case 1: Technical / AI text
    text1 = (
        "Machine learning is a subset of artificial intelligence. "
        "Machine learning algorithms build models from sample data to make predictions. "
        "Deep learning uses neural networks with many layers for complex tasks."
    )
    result1 = keyword_extractor.execute(text=text1, top_n=5)
    print("\n[Test 1] Technical text (top_n=5):")
    print("  Keywords:", [k["word"] for k in result1["keywords"]])
    print("  Metadata:", result1["metadata"])

    # Test case 2: Short text, fewer keywords
    text2 = "Python is great for data science. Python and data science go together."
    result2 = keyword_extractor.execute(text=text2, top_n=3)
    print("\n[Test 2] Short text (top_n=3):")
    print("  Keywords:", [k["word"] for k in result2["keywords"]])

    # Test case 3: Different domain – nature / description
    text3 = (
        "The rainforest is home to countless species. "
        "Rainforest conservation is essential for biodiversity. "
        "Species diversity in the rainforest remains under threat."
    )
    result3 = keyword_extractor.execute(text=text3, top_n=6)
    print("\n[Test 3] Nature/conservation text (top_n=6):")
    print("  Keywords:", [k["word"] for k in result3["keywords"]])

    print()


def run_error_handling() -> None:
    """
    Demonstrate error handling for invalid inputs.

    This section shows how callers should catch exceptions raised by the tool:
    - `ValueError` for invalid values (e.g., empty text, `top_n < 1`)
    - `TypeError` for invalid types (e.g., `text` is not a string)
    """
    print("=" * 60)
    print("ERROR HANDLING – Invalid inputs")
    print("=" * 60)

    # Empty text
    print("\n[Error 1] Empty text:")
    try:
        keyword_extractor.execute(text="")
    except ValueError as e:
        print(f"  Caught ValueError: {e}")

    # Whitespace-only text
    print("\n[Error 2] Whitespace-only text:")
    try:
        keyword_extractor.execute(text="   \n\t  ")
    except ValueError as e:
        print(f"  Caught ValueError: {e}")

    # Negative keyword count
    print("\n[Error 3] Negative top_n:")
    try:
        keyword_extractor.execute(
            text="Some valid text here.",
            top_n=-1,
        )
    except ValueError as e:
        print(f"  Caught ValueError: {e}")

    # Zero keyword count
    print("\n[Error 4] Zero top_n:")
    try:
        keyword_extractor.execute(text="Some valid text.", top_n=0)
    except ValueError as e:
        print(f"  Caught ValueError: {e}")

    # Wrong type for text (TypeError)
    print("\n[Error 5] Invalid type for text (e.g. int):")
    try:
        keyword_extractor.execute(text=123, top_n=5)  # type: ignore[arg-type]
    except TypeError as e:
        print(f"  Caught TypeError: {e}")

    print()


def simple_agent_workflow(input_text: str, max_keywords: int = 5) -> dict:
    """
    Simple agent workflow: process text and extract keywords.

    This function mimics a small “agent step”:
    1) Receive an input string (e.g., a user message or a document).
    2) Call the keyword extraction tool.
    3) Return a consistent, JSON-friendly structure that downstream code can rely on.

    Args:
        input_text: Text to process.
        max_keywords: Maximum number of keywords to request from the tool.

    Returns:
        A dictionary with a stable shape:
        - On success:
          `{\"success\": True, \"keywords\": [str, ...], \"metadata\": {...}}`
        - On failure (validation/type errors):
          `{\"success\": False, \"error\": <message>, \"keywords\": [], \"metadata\": {}}`
    """
    print(f"  [Agent] Processing text ({len(input_text)} chars)...")
    try:
        result = keyword_extractor.execute(text=input_text, top_n=max_keywords)
        keywords = [k["word"] for k in result["keywords"]]
        print(f"  [Agent] Extracted {len(keywords)} keywords: {keywords}")
        return {"success": True, "keywords": keywords, "metadata": result["metadata"]}
    except (ValueError, TypeError) as e:
        print(f"  [Agent] Tool error: {e}")
        return {"success": False, "error": str(e), "keywords": [], "metadata": {}}


def run_agent_workflow_demo() -> None:
    """
    Demonstrate using the keyword extractor inside a simple agent workflow.

    Runs one successful “agent processing” example and one failing example to show how
    the workflow returns structured results without throwing exceptions to callers.
    """
    print("=" * 60)
    print("AGENT WORKFLOW – Process text with keyword extractor")
    print("=" * 60)

    sample = (
        "Natural language processing enables computers to understand human language. "
        "NLP applications include translation, summarization, and sentiment analysis. "
        "Modern NLP relies heavily on transformer models and large language models."
    )

    print("\n[Workflow] Simulated agent processing user query:")
    workflow_result = simple_agent_workflow(sample, max_keywords=5)
    if workflow_result["success"]:
        print(f"  [Agent] Done. Total unique terms: {workflow_result['metadata']['total_terms']}")

    print("\n[Workflow] Agent handling invalid input (empty text):")
    bad_result = simple_agent_workflow("", max_keywords=5)
    print(f"  [Agent] success={bad_result['success']}, error={bad_result.get('error', 'N/A')}")
    print()


def main() -> None:
    """
    Entry point for the demo.

    Runs:
        - successful extraction examples
        - invalid input / error handling examples
        - a simple agent-style workflow example
    """
    print("\nKeyword Extractor Tool – Demo\n")
    run_successful_cases()
    run_error_handling()
    run_agent_workflow_demo()
    print("Demo complete.\n")


if __name__ == "__main__":
    main()
