import sys
import os
import json
import re
import numpy as np

# ==========================================
# 1. LOAD THE word-lm-v1 MODEL
# ==========================================
MODEL_FILE = None
for name in ("word-lm-model.json", "model.json"):
    if os.path.exists(name):
        MODEL_FILE = name
        break

if MODEL_FILE is None:
    print("Error: Could not find word-lm-model.json (or model.json) in the current directory.")
    print("       Export it from the HTML app with '⬇ Export model (.json)'.")
    sys.exit(1)

print(f"Loading {MODEL_FILE}...")
with open(MODEL_FILE, "r") as f:
    data = json.load(f)

if data.get("format") != "word-lm-v1":
    print(f"Error: Expected format 'word-lm-v1', got {data.get('format')!r}")
    sys.exit(1)

vocab       = data["vocab"]
word2id     = {w: i for i, w in enumerate(vocab)}
context_len = int(data["contextLen"])
embed_dim   = int(data["embedDim"])

# Embedding: flat V*D array -> (V, D) row-major
E = np.array(data["embed"], dtype=np.float32).reshape((len(vocab), embed_dim))

# Hidden layers (Dense w/ ReLU, or linear if act != 'relu')
hidden = []
for L in data["hiddenLayers"]:
    W = np.array(L["W"], dtype=np.float32).reshape((int(L["out"]), int(L["in"])))
    b = np.array(L["b"], dtype=np.float32)
    hidden.append({"W": W, "b": b, "act": L["act"]})

# Output layer (sampled softmax head; we run it fully for generation)
outW = np.array(data["output"]["W"], dtype=np.float32).reshape(
    (int(data["output"]["out"]), int(data["output"]["in"])))
outb = np.array(data["output"]["b"], dtype=np.float32)

n_params = E.size + sum(L["W"].size + L["b"].size for L in hidden) + outW.size + outb.size
print(f"Vocab: {len(vocab):,}  |  Context: {context_len}  |  Embed dim: {embed_dim}  |  Params: {n_params:,}")

# ==========================================
# 2. TOKENIZER  (matches the JS app exactly)
# ==========================================
WORD_RE  = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")
PUNCT_RE = re.compile(r"^[^A-Za-z0-9_\s]$")   # single non-word, non-space char

def tokenize(text):
    return WORD_RE.findall(text.lower())

def join_tokens(tokens):
    out = []
    for i, t in enumerate(tokens):
        if i == 0 or PUNCT_RE.match(t):
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)

# ==========================================
# 3. FORWARD PASS
# ==========================================
def forward(ids):
    # ids: np.int32 array of shape (context_len,)
    # Embedding lookup + concat  ->  (context_len * embed_dim,)
    a = E[ids].reshape(-1)

    for L in hidden:
        z = L["W"] @ a + L["b"]
        a = np.maximum(z, 0) if L["act"] == "relu" else z

    return outW @ a + outb

# ==========================================
# 4. SAMPLING (temp + nucleus)
# ==========================================
def sample_next(ids, temp=0.8, top_p=0.92):
    logits = forward(ids) / max(temp, 0.05)

    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)

    order = np.argsort(probs)[::-1]
    sp    = probs[order]
    cum   = np.cumsum(sp)

    cutoff = int(np.searchsorted(cum, top_p)) + 1
    cutoff = min(cutoff, len(sp))

    order = order[:cutoff]
    sp    = sp[:cutoff]
    sp   /= np.sum(sp)

    return int(np.random.choice(order, p=sp))

# ==========================================
# 5. MAIN
# ==========================================
def main():
    length = 100
    temp   = 0.8
    top_p  = 0.92

    if len(sys.argv) == 1:
        try:
            prompt = input(">>> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
    else:
        prompt = sys.argv[1]
        if len(sys.argv) > 2: length = int(sys.argv[2])
        if len(sys.argv) > 3: temp   = float(sys.argv[3])
        if len(sys.argv) > 4: top_p  = float(sys.argv[4])

    tokens = tokenize(prompt)
    ids    = [word2id.get(t, 0) for t in tokens]   # OOV -> <unk> (id 0)

    oov = sum(1 for t in tokens if t not in word2id)
    if oov:
        print(f"\033[90m({oov}/{len(tokens)} prompt tokens are OOV -> <unk>)\033[0m")

    # Pad (left) with <unk>, or truncate to context_len
    while len(ids) < context_len:
        ids.insert(0, 0)
    if len(ids) > context_len:
        ids = ids[-context_len:]

    context = np.array(ids, dtype=np.int32)

    # Echo prompt in gray
    sys.stdout.write(f"\033[90m{prompt}\033[0m")
    sys.stdout.flush()

    generated = []
    written   = 0
    try:
        for _ in range(length):
            nxt = sample_next(context, temp, top_p)
            generated.append(vocab[nxt])

            # Print only the newly-joined delta in green
            text = join_tokens(generated)
            sys.stdout.write(f"\033[92m{text[written:]}\033[0m")
            sys.stdout.flush()
            written = len(text)

            context = np.roll(context, -1)
            context[-1] = nxt
    except KeyboardInterrupt:
        pass

    print("\n\n\033[90m[Generation stopped]\033[0m\n")


if __name__ == "__main__":
    main()