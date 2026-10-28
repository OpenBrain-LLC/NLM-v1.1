# NLM-1.1 (Neural Language Model)

A word-level language model that trains and runs locally trained entirely with vanilla JavaScript and Python/NumPy (for the CLI inference runner).

### Specs
*   **Architecture:** Feed-Forward Neural Network (Sparse-Input MLP)
*   **Parameters:** ~375,000
*   **Vocab:** ~2,000 tokens (word-level)
*   **Context Window:** 8 tokens
*   **Embeddings:** 32 dims
*   **Hidden Layers:** 2 × 128 (ReLU)
*   **File Size:** ~1.5 MB (raw JSON)

### How to run it
You need `__main__.py` and `model.json`.

```bash
pip install -r requirements.txt
python __main__.py
```

Or use the shell script:
```bash
./app.sh
```

### Sample Output
Because the context window is only 8 tokens (roughly one short phrase), the model generates perfect local grammar but completely loses the plot over longer sequences. 

**Prompt:** `The`
**Output:**
> `The house, one the contractor was the new and trader. had became the first battle, william long the formed 14 town) the maryland governor of the cadden of hubbardton whom the henry they the house... executives= sandwich withstand(`

### The History of NLM
*   **V1.0 (148K Params):** Character-level. Generated phonetic babble (`the sowe the rowes oun ald be che ther`).
*   **V1.1 (374K Params; now):** Word-level. Generates syntactically valid, grammatically formatted nonsense (`executives= sandwich withstand(`).
