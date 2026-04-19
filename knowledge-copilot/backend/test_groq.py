"""
Quick test: confirms Groq key is valid and Llama 3.1 70B responds.

Run from backend/ with venv active:
    python test_groq.py
"""
import os, time, sys

# ── 1. Check key exists ───────────────────────────────────────────────────────
key = os.getenv("GROQ_API_KEY") or ""

# Try loading from .env manually if not in environment
if not key:
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass

if not key or key == "gsk_your_key_here":
    print("✗ GROQ_API_KEY not set in .env")
    print("  1. Go to console.groq.com")
    print("  2. Create an API key")
    print("  3. Add GROQ_API_KEY=gsk_... to backend/.env")
    sys.exit(1)

print(f"✓ Key found: {key[:12]}...")

# ── 2. Test direct Groq SDK ───────────────────────────────────────────────────
try:
    from groq import Groq
    client = Groq(api_key=key)

    print("\nTesting Groq API directly...")
    t0 = time.time()
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Groq is working' and nothing else."},
        ],
        max_tokens=20,
        temperature=0,
    )
    elapsed = time.time() - t0
    answer  = resp.choices[0].message.content
    tps     = resp.usage.completion_tokens / elapsed

    print(f"  Response : {answer}")
    print(f"  Latency  : {elapsed*1000:.0f}ms")
    print(f"  Speed    : ~{tps:.0f} tokens/sec")
    print(f"  Model    : {resp.model}")
    print("✓ Groq SDK working\n")
except Exception as e:
    print(f"✗ Groq SDK error: {e}")
    sys.exit(1)

# ── 3. Test LangChain integration ─────────────────────────────────────────────
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=key,
        temperature=0,
        max_tokens=50,
    )

    print("Testing LangChain + Groq...")
    t0 = time.time()
    response = llm.invoke([
        SystemMessage(content="You are a RAG assistant. Answer only from context."),
        HumanMessage(content="Context: The sky is blue. Question: What colour is the sky?"),
    ])
    elapsed = time.time() - t0

    print(f"  Response : {response.content}")
    print(f"  Latency  : {elapsed*1000:.0f}ms")
    print("✓ LangChain + Groq working\n")
except Exception as e:
    print(f"✗ LangChain error: {e}")
    sys.exit(1)

# ── 4. Test streaming ─────────────────────────────────────────────────────────
try:
    print("Testing streaming...")
    llm_stream = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=key,
        temperature=0,
        max_tokens=60,
    )

    t0     = time.time()
    tokens = []
    print("  Stream: ", end="", flush=True)
    for chunk in llm_stream.stream([HumanMessage(content="Count to 5, one number per word.")]):
        if chunk.content:
            tokens.append(chunk.content)
            print(chunk.content, end="", flush=True)
    elapsed = time.time() - t0
    print()
    print(f"  Latency  : {elapsed*1000:.0f}ms total, {len(tokens)} chunks")
    print("✓ Streaming working\n")
except Exception as e:
    print(f"✗ Streaming error: {e}")
    sys.exit(1)

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 50)
print("All checks passed. Groq + Llama 3.1 70B is ready.")
print("Start your server: python main.py")
print("=" * 50)