import numpy as np
from app.services.embedder import embed_query

sentences = [
    "How do neural networks learn?",
    "What is backpropagation?",        # similar to above
    "Best restaurants in Paris",       # completely different
]

vectors = [embed_query(s) for s in sentences]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"ML pair similarity:    {cosine_similarity(vectors[0], vectors[1]):.4f}")  # expect ~0.75+
print(f"Unrelated similarity:  {cosine_similarity(vectors[0], vectors[2]):.4f}")  # expect ~0.10-0.30