# Fine-Tuning GPT-2 for Text Generation — Complete Explanation

---

## 1. What is GPT-2?

GPT-2 stands for **Generative Pre-trained Transformer 2**. It was created by OpenAI in 2019.

- It is a **language model** — a model that understands and generates human language
- It was pre-trained on **40GB of internet text** (news, books, websites)
- Its job is simple: **given some text, predict what comes next**
- It uses a **Transformer architecture** — specifically a "decoder-only" transformer with self-attention layers

### GPT-2 Variants

| Model | Parameters | Speed | Quality |
|---|---|---|---|
| GPT-2 Small | 117 Million | Fastest | Basic |
| GPT-2 Medium | 345 Million | Fast | Good |
| GPT-2 Large | 774 Million | Slow | Better |
| GPT-2 XL | 1.5 Billion | Slowest | Best |

> We use **GPT-2 Small** in this task — it's the lightest and works fine for learning.

---

## 2. What is Fine-Tuning?

GPT-2 already knows how to write English — but it writes in a **general style**.

**Fine-tuning** means continuing to train GPT-2 on **your specific dataset**, so it learns to write in your style or domain.

### Analogy
> Think of GPT-2 as a student who has read millions of books. Fine-tuning is like giving that student a specialised course — say, medical journals or movie scripts — so they start writing in that specific style.

### Why Fine-Tune Instead of Train from Scratch?
- Training from scratch needs **billions of words** and weeks of compute
- Fine-tuning needs only **a few hundred to thousand sentences**
- It is **much faster and cheaper**
- The model already knows grammar, facts, and language — you're just redirecting its style

---

## 3. How Does GPT-2 Read Text? — Tokenization

Computers can't read words directly. GPT-2 converts text into **tokens** (numbered chunks).

### Example
```
Input  : "Artificial intelligence is transforming the world."
Tokens : ['Art', 'ificial', ' intelligence', ' is', ' transform', 'ing', ' the', ' world', '.']
IDs    : [आर्ट, 9542, 4430, 318, 25449, 278, 262, 995, 13]
```

- Each word (or part of a word) becomes a number
- GPT-2 has a vocabulary of **50,257 tokens**
- The model reads and predicts at the **token level**, not the word level

---

## 4. The Training Process

### Step-by-Step

```
Your Text File
      ↓
  Tokenization  →  Convert text to token IDs
      ↓
  Block Splitting  →  Divide into chunks of 128 tokens
      ↓
  Forward Pass  →  Model predicts next token at each position
      ↓
  Loss Calculation  →  Compare predictions vs actual tokens
      ↓
  Backpropagation  →  Adjust model weights to reduce loss
      ↓
  Repeat for N Epochs
      ↓
  Fine-Tuned Model ✅
```

### Key Terms

| Term | Meaning |
|---|---|
| **Epoch** | One full pass through the entire dataset |
| **Batch size** | How many examples the model sees at once |
| **Learning rate** | How big each weight update step is |
| **Loss** | A number measuring how wrong the model's predictions are |
| **Backpropagation** | The algorithm that updates weights to reduce loss |

### What is Loss?
- Loss starts high (model is making many wrong predictions)
- As training progresses, loss goes **down** — the model is learning
- A **falling loss curve** = successful training

---

## 5. Text Generation Strategies

After training, you give GPT-2 a **prompt** and it generates a continuation. There are different ways to choose the next token:

### Strategy 1 — Greedy Decoding
- Always picks the **single most likely** next token
- **Pros:** Fast, predictable
- **Cons:** Repetitive and boring
- Example output: *"AI is the future. AI is the future. AI is the future."*

### Strategy 2 — Beam Search
- Explores **multiple paths** simultaneously (like a chess engine)
- Keeps the top N sequences at each step
- **Pros:** More coherent and grammatically correct
- **Cons:** Can still be repetitive; not very creative

### Strategy 3 — Top-K Sampling
- At each step, considers only the **top K most likely tokens** and samples randomly among them
- K=50 means: pick randomly from the 50 most probable next tokens
- **Pros:** More variety and creativity
- **Cons:** Can occasionally pick weird tokens

### Strategy 4 — Top-P (Nucleus) Sampling
- Instead of a fixed K, picks from the **smallest set of tokens whose probabilities sum to P**
- P=0.92 means: keep adding tokens until their combined probability reaches 92%
- **Pros:** Dynamically adapts — uses more tokens when uncertain, fewer when confident
- **Cons:** Slightly more complex to tune

### Strategy 5 — Temperature
- A multiplier that controls **how random** the sampling is
- Applied alongside Top-K or Top-P

| Temperature | Effect |
|---|---|
| 0.1 – 0.5 | Very focused, conservative, repetitive |
| 0.7 – 0.9 | Balanced — good quality + some variety |
| 1.0 | Neutral (no change) |
| 1.2 – 1.5 | Very creative, sometimes incoherent |

### ⭐ Best Practice
Use **Top-K + Top-P + Temperature together** for the best quality output.

---

## 6. Evaluation — Perplexity

**Perplexity (PPL)** measures how well your model predicts text.

- Think of it as: *"How surprised is the model by this text?"*
- **Lower perplexity = better model** (less surprised = more confident)
- Formula: `PPL = exp(average cross-entropy loss)`

### Interpretation

| Perplexity | Meaning |
|---|---|
| Very low (< 20) | Model knows this domain very well |
| Medium (20–100) | Reasonable performance |
| High (> 200) | Model is struggling / needs more training |

After fine-tuning, your model's perplexity on your domain text should be **lower** than the base GPT-2.

---

## 7. The HuggingFace `Trainer` API

Instead of writing the training loop manually, we use HuggingFace's **`Trainer`** class.

It handles:
- Loading data in batches
- Running forward and backward passes
- Logging loss at each step
- Saving model checkpoints
- GPU/CPU management

You just configure `TrainingArguments` and call `trainer.train()`.

---

## 8. End-to-End Flow Summary

```
1. Install libraries        →  transformers, datasets, torch
2. Prepare dataset          →  Your custom text file
3. Load GPT-2               →  Pre-trained weights from HuggingFace
4. Tokenize dataset         →  Convert text → token ID blocks
5. Configure training       →  Epochs, batch size, learning rate
6. Fine-tune                →  trainer.train()
7. Save model               →  trainer.save_model()
8. Generate text            →  model.generate() with your prompt
9. Evaluate                 →  Compute Perplexity (lower = better)
```

---

## 9. Tips for Better Results

| Tip | Details |
|---|---|
| **Use more data** | Aim for at least 10,000+ words in your dataset |
| **Train more epochs** | Try 10–30 epochs for small datasets |
| **Use a GPU** | Training on CPU is 10–50x slower |
| **Avoid overfitting** | If loss reaches 0 too fast, model memorised the data — add more variety |
| **Use a larger model** | GPT-2 Medium/Large gives better quality |
| **Clean your data** | Remove duplicates, HTML tags, and noise |

---

## 10. Glossary

| Term | Definition |
|---|---|
| **Transformer** | Neural network architecture using self-attention |
| **Token** | A chunk of text (word or subword) converted to a number |
| **Epoch** | One complete training pass over all data |
| **Loss** | Error metric — lower is better |
| **Perplexity** | How surprised the model is by text — lower is better |
| **Fine-tuning** | Further training a pre-trained model on new data |
| **Top-K / Top-P** | Sampling strategies that control text diversity |
| **Temperature** | Controls randomness during text generation |
| **HuggingFace** | Python library providing pre-trained models and training tools |
