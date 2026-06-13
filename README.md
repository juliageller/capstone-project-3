# Automated Dialogue Summarization with BART

Project 3 - Large Language Models
Katherine Geller

This project fine-tunes a BART model to summarize chat conversations, using the SAMSum dataset. The notebook `dialogue_summarization.ipynb` has the full code and walks through everything step by step.

## Problem statement and business context

Messaging apps (Slack, Teams, WhatsApp, Discord) generate a ton of messages every day, and people miss important things buried in long threads. A knowledge worker spends roughly a quarter of the work week on messaging, and catching up on a busy channel can take 20-40 minutes.

The idea is a feature that takes a long group chat and produces a short summary so someone can catch up in a few seconds. This makes the platform more useful day to day and opens up paid-tier options (longer context, custom summary styles, etc.).

## Technical approach and methodology

- **Dataset:** SAMSum (~16k messenger-style conversations with human summaries), loaded from Hugging Face.
- **Model:** `facebook/bart-base`, a pretrained encoder-decoder transformer. I fine-tuned it instead of training from scratch because SAMSum is too small to train a good model from nothing, and BART already knows a lot of language from pretraining.
- **Why encoder-decoder:** summarization is a sequence-to-sequence task (text in, different text out). The encoder reads the whole conversation and the decoder writes the summary one word at a time. A plain BERT (encoder only) can't generate text, so it wouldn't work here.
- **Training:** Hugging Face `Seq2SeqTrainer`, 3 epochs, batch size 8, learning rate 5e-5, with fp16 mixed precision when a GPU is available.
- **Generation:** beam search (4 beams) with no-repeat-ngram to avoid repeating phrases.

## Results and evaluation

I evaluated on the held-out test set using ROUGE, BERTScore, and a small human review.

**Automatic metrics (test set):**

| Metric | My model | Pitch target |
|---|---|---|
| ROUGE-1 | 50.77 | 45 |
| ROUGE-2 | 25.78 | 20 |
| ROUGE-L | 42.02 | 40 |
| BERTScore F1 | 0.921 | 0.88 |

The model beat all four targets. ROUGE measures word overlap with the human summary (the standard SAMSum metric), and BERTScore measures meaning rather than exact words.

**Human review (15 samples, scored 1-5):**

| Dimension | Avg score |
|---|---|
| Fluency | 4.93 |
| Conciseness | 4.87 |
| Informativeness | 3.87 |
| Faithfulness | 3.73 |

The summaries read very cleanly (high fluency/conciseness). The weakest area was faithfulness - on a few multi-person chats the model shifted who said what or added a small detail that wasn't there. This matches the limitations below.

## Discussion of limitations and future work

**Limitations:**
- BART can only read 1024 tokens, so very long conversations get cut off and details near the end can be missed.
- The model occasionally adds a small detail that wasn't in the chat, or shifts who said what (the faithfulness issue seen in the human review, ~3.73/5).
- Only trained and tested on English.

**Future work:**
- Try a bigger model (BART-large or T5-base) to push the scores up.
- Use a long-document model (Longformer/LED) for the really long chats.
- More tuning of the beam search settings.
- For a real product I'd want a human-in-the-loop or a disclaimer because of the hallucination risk.

## How to run the code

I don't have a strong GPU on my own machine, so I run this on **Google Colab** for free. Steps:

1. Go to https://colab.research.google.com and sign in with a Google account.
2. `File > Upload notebook` and choose `dialogue_summarization.ipynb`.
3. `Runtime > Change runtime type`, pick **T4 GPU**, and Save.
4. `Runtime > Run all`. Training takes about a couple of hours, so just leave it.

Kaggle Notebooks also works (enable the GPU in the settings panel). The notebook will still run on a laptop (Apple Silicon or CPU) thanks to a device fallback, but training is very slow there, so I'd only use a laptop for inference.

The first code cell installs the few libraries that Colab doesn't already include. If you'd rather install locally, see `requirements.txt`.

## Files in this repo

- `dialogue_summarization.ipynb` - the full notebook (code + explanations + results)
- `README.md` - this file
- `requirements.txt` - Python packages needed

## Model weights

The fine-tuned model is saved by the notebook to the `bart-samsum-final/` folder. It's a few hundred MB, so I didn't commit it to the repo. To recreate it, just run the notebook end to end - the saved folder can be reloaded with `AutoModelForSeq2SeqLM.from_pretrained("bart-samsum-final")` without retraining.

## References

- SAMSum dataset: Gliwa et al., "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization" (2019)
- BART: Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training..." (2019)
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
