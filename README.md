# Text Summarization Using Transformer Models: T5 and BART on CNN/DailyMail Dataset

This repository contains code for training and evaluating transformer-based models for text summarization. The primary goal is to utilize pre-trained models such as T5 and BART to summarize text from the CNN/DailyMail dataset.

## Table of Contents
- [Setup and Requirements](#setup-and-requirements)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Setup and Requirements

The following Python libraries are required to run the code:

```bash
pip install transformers datasets evaluate rouge_score
```

## Dataset

The code uses the CNN/DailyMail dataset (version 3.0.0) for training and evaluation. A subset of 1000 samples is used for demonstration purposes. The dataset is split into training and testing sets with an 80-20 ratio.

## Preprocessing

- Articles are prefixed with the string `"summarize: "`.
- Text is tokenized using the tokenizer corresponding to the chosen pre-trained model (T5 or BART).
- Highlights are tokenized as labels for training.
- The tokenized data is prepared with a maximum input length of 1024 tokens and label length of 128 tokens.

## Model Training

- Two models are used: `t5-small` and `facebook/bart-base`.
- Training is configured using `Seq2SeqTrainer` from the Hugging Face Transformers library.
- Training arguments include:
  - Batch size: 16
  - Learning rate: 2e-5
  - Weight decay: 0.01
  - Number of epochs: 4
  - Mixed precision (fp16) enabled
  - Checkpoint saving limited to the last three models

## Evaluation

Evaluation is performed using the ROUGE metric:
- Computes ROUGE scores between generated summaries and reference highlights.
- Average generation length (`gen_len`) is also calculated.

## Usage

1. **Train the Model:**
   ```python
   trainer.train()
   ```

2. **Generate Summaries:**
   Example usage for generating summaries from text:
   ```python
   inputs = tokenizer(text, return_tensors="pt").input_ids
   outputs = model.generate(inputs.cuda(), max_new_tokens=100, do_sample=False)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

## Results

Both T5 and BART models were trained and evaluated on the dataset. Evaluation metrics (ROUGE scores) and generated summaries demonstrate the capability of these models to perform text summarization tasks effectively.

## Acknowledgments

The code is based on the Hugging Face Transformers library and uses the CNN/DailyMail dataset. Special thanks to the Hugging Face team for providing pre-trained models and tools for fine-tuning.

---

### Notes
- Ensure proper GPU setup for efficient training.
- Use `huggingface-cli login` to enable model saving to the Hugging Face Hub.

