"""
Kimi-K2.6 client examples — OpenAI-compatible API via SGLang.

Prerequisites:
    pip install openai
    # Access requires a running SLURM serve job

Usage:
    sbatch serve.sh
    squeue -j <jobid> -o "%N"
    ssh -N -L 8000:<compute-node>:8000 <login-node>
    python client_example.py
"""

import time

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"

client = OpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=3600)


def check_server():
    """Verify the server is running and model is loaded."""
    models = client.models.list()
    print(f"Available models: {[m.id for m in models.data]}")


def chat_thinking(prompt: str) -> str:
    """Chat with thinking mode (default). Returns reasoning + response."""
    start = time.time()
    response = client.chat.completions.create(
        model="Kimi-K2.6",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
    )
    elapsed = time.time() - start
    msg = response.choices[0].message
    print(f"[Thinking mode] {elapsed:.1f}s")
    if hasattr(msg, "reasoning") and msg.reasoning:
        print(f"  Reasoning: {msg.reasoning[:200]}...")
    print(f"  Response: {msg.content}")
    return msg.content


def chat_instant(prompt: str) -> str:
    """Chat with instant mode (no thinking). Faster, less reasoning."""
    start = time.time()
    response = client.chat.completions.create(
        model="Kimi-K2.6",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
    elapsed = time.time() - start
    msg = response.choices[0].message
    print(f"[Instant mode] {elapsed:.1f}s")
    print(f"  Response: {msg.content}")
    return msg.content


def chat_with_tools():
    """Demonstrate tool calling."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_plasma_current",
                "description": "Get the plasma current for a given tokamak shot number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shot_number": {
                            "type": "integer",
                            "description": "The tokamak shot/pulse number",
                        },
                        "device": {
                            "type": "string",
                            "enum": ["ITER", "JET", "TCV", "WEST"],
                            "description": "The tokamak device name",
                        },
                    },
                    "required": ["shot_number", "device"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="Kimi-K2.6",
        messages=[
            {
                "role": "user",
                "content": "What was the plasma current for ITER shot 100001?",
            }
        ],
        tools=tools,
        max_tokens=4096,
    )
    msg = response.choices[0].message
    print("[Tool calling]")
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  Tool: {tc.function.name}({tc.function.arguments})")
    else:
        print(f"  Response: {msg.content}")


if __name__ == "__main__":
    print("=" * 60)
    print("Kimi-K2.6 Client Examples")
    print("=" * 60)

    print("\n1. Server check")
    check_server()

    print("\n2. Thinking mode (default)")
    chat_thinking("Which is bigger, 9.11 or 9.9? Think carefully.")

    print("\n3. Instant mode (no thinking)")
    chat_instant("Explain tokamak plasma confinement in one paragraph.")

    print("\n4. Tool calling")
    chat_with_tools()
