# PRODIGY_GA_01
Fine-tuning GPT-2 for custom text generation

## 📌 Task
Fine-tune GPT-2 to generate coherent and contextually relevant text based on a given prompt.

## 📋 Description
This project is part of my Generative AI Internship at Prodigy InfoTech. I have fine-tuned OpenAI's GPT-2 
transformer model on a custom dataset to generate domain-specific text that mimics the style and 
structure of the training data. The project covers the complete pipeline from data preparation to 
text generation using multiple decoding strategies.

## 🛠️ Tech Stack
- Python 3.12
- HuggingFace Transformers
- PyTorch
- Google Colab (GPU)
- Matplotlib

## 📂 Files
| File | Description |
|---|---|
| `gpt2_finetune.py` | Complete Python code for fine-tuning and text generation |
| `GPT2_Explanation.md` | Detailed explanation of all concepts used in this project |
| `GPT2_Finetuning_TextGeneration.ipynb` | Jupyter notebook version — run directly in Google Colab |
| `requirements.txt` | All required Python libraries with versions |

## 📊 Results
| Metric | Value |
|---|---|
| Training Loss (start) | 3.4056 |
| Training Loss (end) | 2.3155 |
| Base GPT-2 Perplexity | 35.13 |
| Fine-tuned GPT-2 Perplexity | 9.86 |
| Improvement | 71.9% ✅ |

## 🔮 Text Generation Strategies Implemented
| Strategy | Description |
|---|---|
| Greedy Decoding | Always picks the most likely next token |
| Beam Search | Explores multiple sequences simultaneously |
| Top-K Sampling | Samples from top K most probable tokens |
| Top-P (Nucleus) Sampling | Samples from tokens covering top P% probability |
| Combined Strategy | Top-K + Top-P + Temperature (best results) |

## 💬 Sample Output
**Prompt:** `Artificial intelligence is`

**Combined Strategy Output:**
```
Artificial intelligence is transforming the way we live and work.
At its core, AI programs can now do nearly anything from translating 
text into image or music to designing complex models of natural 
language processing networks.
```

## ▶️ How to Run
1. Open `GPT2_Finetuning_TextGeneration.ipynb` in Google Colab
2. Set Runtime to GPU — Runtime → Change runtime type → T4 GPU
3. Run Cell 1 to install dependencies:
```
!pip install -q accelerate==0.30.0 peft==0.10.0
```
4. Restart runtime after installation
5. Run Cell 2 with the main code

## 📚 References
- [HuggingFace — How to Generate Text](https://huggingface.co/blog/how-to-generate)
- [GPT-2 Fine-Tuning Colab Notebook](https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD)

