# =============================================================
#   GPT-2 Fine-Tuning for Text Generation
#   Run this in Google Colab (Runtime → GPU recommended)
# =============================================================

# ── STEP 1: INSTALL LIBRARIES ─────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "transformers==4.35.0", "datasets==2.14.6",
                "accelerate==0.24.0", "sentencepiece", "-q"], check=True)

# ── STEP 2: IMPORTS ───────────────────────────────────────────
import os
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset

# Reproducibility
set_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ── STEP 3: CONFIGURATION ─────────────────────────────────────
# ✏️ Change these settings as needed

MODEL_NAME      = "gpt2"          # Options: "gpt2", "gpt2-medium", "gpt2-large"
OUTPUT_DIR      = "./gpt2-finetuned"
TRAIN_FILE      = "train_data.txt"
BLOCK_SIZE      = 128             # Token sequence length per training sample
NUM_EPOCHS      = 5               # Increase for better results (try 10–20)
BATCH_SIZE      = 4               # Reduce to 2 if you run out of memory
LEARNING_RATE   = 5e-5
MAX_NEW_TOKENS  = 80              # How many tokens to generate


# ── STEP 4: PREPARE DATASET ───────────────────────────────────
# ✏️ REPLACE THIS TEXT WITH YOUR OWN DATASET
# Paste in your own articles, stories, reviews, etc.

custom_text = """
Artificial intelligence is transforming the way we live and work.
Machine learning algorithms can now recognize faces, translate languages, and even write code.
Deep learning, a subset of machine learning, uses neural networks with many layers to learn from data.
These networks are inspired by the structure of the human brain.
Natural language processing allows computers to understand and generate human language.
Transformer models like GPT-2 have revolutionized the field of text generation.
They use attention mechanisms to capture long-range dependencies in text.
Fine-tuning allows us to adapt these powerful models to specific tasks and domains.
The future of AI holds great promise for solving complex problems in healthcare, education, and science.
Researchers continue to push the boundaries of what machines can learn and do.
Ethical AI development requires careful consideration of bias, fairness, and transparency.
Data privacy and security are critical concerns in the age of big data and machine learning.
Reinforcement learning enables agents to learn through trial and error in simulated environments.
Computer vision systems can now surpass human performance on specific image recognition tasks.
The collaboration between humans and AI systems is becoming increasingly important.
Generative models can create realistic images, text, music, and even video.
Transfer learning allows models trained on one task to be applied to related tasks.
The democratization of AI tools is enabling more people to build intelligent applications.
Open-source frameworks like PyTorch and TensorFlow have accelerated AI research.
Cloud computing provides the infrastructure needed to train large-scale AI models.
"""

# Write dataset to file (repeated to create more training samples)
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for _ in range(10):
        f.write(custom_text.strip() + "\n")

with open(TRAIN_FILE, "r") as f:
    content = f.read()

print(f"Dataset saved: {len(content.split())} words, {len(content.splitlines())} lines")


# ── STEP 5: LOAD TOKENIZER AND MODEL ──────────────────────────

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token by default

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME}  |  Parameters: {total_params:,}")


# ── STEP 6: TOKENIZE DATASET ──────────────────────────────────

def build_dataset(file_path, tokenizer, block_size):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )["input_ids"][0]

    # Split into fixed-size blocks
    blocks = [
        token_ids[i : i + block_size].tolist()
        for i in range(0, len(token_ids) - block_size + 1, block_size)
    ]

    return Dataset.from_dict({"input_ids": blocks, "labels": blocks})


train_dataset = build_dataset(TRAIN_FILE, tokenizer, BLOCK_SIZE)
print(f"Training blocks: {len(train_dataset)}  (block size = {BLOCK_SIZE} tokens)")


# ── STEP 7: FINE-TUNE ─────────────────────────────────────────

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    seed=42,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

print("Starting fine-tuning...")
train_result = trainer.train()

print(f"\nFine-tuning complete!")
print(f"  Training time : {train_result.metrics['train_runtime']:.1f} seconds")
print(f"  Final loss    : {train_result.metrics['train_loss']:.4f}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to '{OUTPUT_DIR}'")


# ── STEP 8: PLOT TRAINING LOSS ────────────────────────────────

log_history = trainer.state.log_history
train_losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e]

if train_losses:
    steps, losses = zip(*train_losses)
    plt.figure(figsize=(9, 4))
    plt.plot(steps, losses, marker="o", color="royalblue", linewidth=2, markersize=4)
    plt.fill_between(steps, losses, alpha=0.15, color="royalblue")
    plt.title("Training Loss over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=100)
    plt.show()
    print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")


# ── STEP 9: LOAD FINE-TUNED MODEL ─────────────────────────────

ft_tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
ft_tokenizer.pad_token = ft_tokenizer.eos_token

ft_model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR).to(DEVICE)
ft_model.eval()

print("Fine-tuned model loaded and ready.")


# ── STEP 10: TEXT GENERATION FUNCTION ─────────────────────────

def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.85,
    top_k=50,
    top_p=0.95,
    num_sequences=1,
):
    """
    Generate text using the fine-tuned GPT-2 model.

    Args:
        prompt         : Starting text for generation
        max_new_tokens : Number of new tokens to generate
        temperature    : Randomness — low (0.3) = focused, high (1.3) = creative
        top_k          : Sample from top K probable tokens
        top_p          : Sample from tokens covering top P% probability
        num_sequences  : How many different outputs to return
    """
    input_ids = ft_tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = ft_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            num_return_sequences=num_sequences,
            pad_token_id=ft_tokenizer.eos_token_id,
        )

    return [ft_tokenizer.decode(o, skip_special_tokens=True) for o in output]


# ── STEP 11: DECODING STRATEGY COMPARISON ─────────────────────

PROMPT = "Artificial intelligence is"   # ✏️ Change this prompt!

input_ids = ft_tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)

print(f"\nPrompt: '{PROMPT}'\n")

# 1. Greedy Decoding
with torch.no_grad():
    out = ft_model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
print("── GREEDY DECODING ──────────────────────────────────────")
print(ft_tokenizer.decode(out[0], skip_special_tokens=True))

# 2. Beam Search
with torch.no_grad():
    out = ft_model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS,
                             num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
print("\n── BEAM SEARCH (beams=5) ───────────────────────────────")
print(ft_tokenizer.decode(out[0], skip_special_tokens=True))

# 3. Top-K Sampling
with torch.no_grad():
    out = ft_model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=True, top_k=50, temperature=0.8)
print("\n── TOP-K SAMPLING (k=50, temp=0.8) ────────────────────")
print(ft_tokenizer.decode(out[0], skip_special_tokens=True))

# 4. Top-P (Nucleus) Sampling
with torch.no_grad():
    out = ft_model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=True, top_p=0.92, top_k=0, temperature=0.9)
print("\n── TOP-P NUCLEUS SAMPLING (p=0.92, temp=0.9) ──────────")
print(ft_tokenizer.decode(out[0], skip_special_tokens=True))

# 5. Best Combined Strategy
with torch.no_grad():
    out = ft_model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.85,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        num_return_sequences=3,
    )
print("\n── COMBINED: TOP-K + TOP-P + TEMP (3 samples) ─────────")
for i, o in enumerate(out, 1):
    print(f"\n[Sample {i}] {ft_tokenizer.decode(o, skip_special_tokens=True)}")


# ── STEP 12: PERPLEXITY EVALUATION ────────────────────────────

def compute_perplexity(model, tokenizer, text, device, max_length=512):
    """Lower perplexity = better model."""
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return math.exp(loss.item())


test_text = (
    "Machine learning algorithms can now recognize faces and translate languages. "
    "Deep learning uses neural networks with many layers. "
    "The future of artificial intelligence is full of possibilities."
)

base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_tok   = GPT2Tokenizer.from_pretrained("gpt2")
base_tok.pad_token = base_tok.eos_token

ppl_base = compute_perplexity(base_model, base_tok,   test_text, DEVICE)
ppl_ft   = compute_perplexity(ft_model,   ft_tokenizer, test_text, DEVICE)

print(f"\nPerplexity — Base GPT-2      : {ppl_base:.2f}")
print(f"Perplexity — Fine-tuned GPT-2: {ppl_ft:.2f}")

if ppl_ft < ppl_base:
    print(f"✅ Improvement: {((ppl_base - ppl_ft) / ppl_base) * 100:.1f}%")
else:
    print("⚠️  Try more epochs or a larger dataset for better results.")

# Bar chart
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Base GPT-2", "Fine-tuned"], [ppl_base, ppl_ft],
               color=["#e07b54", "#4a90d9"], width=0.4)
ax.set_ylabel("Perplexity (lower is better)")
ax.set_title("Model Perplexity Comparison")
for bar, val in zip(bars, [ppl_base, ppl_ft]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("perplexity_comparison.png", dpi=100)
plt.show()


# ── STEP 13: TEMPERATURE EXPERIMENT ───────────────────────────

prompt = "Researchers in artificial intelligence"
print(f"\nTemperature Experiment | Prompt: '{prompt}'\n")

for temp in [0.3, 0.7, 1.0, 1.3]:
    result = generate_text(prompt, max_new_tokens=60, temperature=temp)
    print(f"[Temperature = {temp}]")
    print(result[0])
    print()


# ── STEP 14: GENERATE WITH YOUR OWN PROMPTS ───────────────────
# ✏️ Add or change any prompts here!

my_prompts = [
    "Deep learning is",
    "The future of AI in healthcare",
    "Neural networks can",
]

print("\nCustom Prompt Generation\n" + "=" * 55)
for prompt in my_prompts:
    print(f"\nPrompt: '{prompt}'")
    results = generate_text(prompt, max_new_tokens=80, temperature=0.85)
    print(results[0])

print("\n" + "=" * 55)
print("  Done! GPT-2 fine-tuning and generation complete.")
print("=" * 55)
