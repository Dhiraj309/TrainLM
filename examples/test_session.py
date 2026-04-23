from pathlib import Path
import time

from aipm.registry.store import Store
from aipm.runtime.loader import load_installed_capabilities
from aipm.adapters.transformers import TransformersAdapter
from aipm.runtime.session import Session


def main():
    print("\n=== AIPM TEST SESSION ===")

    # --- Step 1: Load capabilities ---
    store = Store()
    capabilities = load_installed_capabilities(store)

    print(f"\nLoaded capabilities: {len(capabilities)}")

    if not capabilities:
        print("❌ No capabilities installed.")
        print("Run: aipm add packages/builtin/http_fetch")
        return

    for cap in capabilities:
        print(f" - {cap.name}@{cap.version}")

    # --- Step 2: Initialize model ---
    print("\nLoading model...")

    adapter = TransformersAdapter(
        model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        device="cpu",  # change to "cuda" if available
        max_new_tokens=128,
        temperature=0.2,
    )

    print("Model loaded.")

    # --- Step 3: Create session ---
    session = Session(
        adapter=adapter,
        capabilities=capabilities,
        base_path=store.packages,
        max_steps=5,
    )

    # --- Step 4: Run query ---
    query = "Fetch https://example.com"

    print("\n=== USER ===")
    print(query)

    print("\n=== RUNNING ===")

    start_time = time.time()

    response = session.run(query)

    end_time = time.time()

    # --- Step 5: Output ---
    print("\n=== FINAL RESPONSE ===")
    print(response)

    print(f"\n=== TIME ===\n{end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()
