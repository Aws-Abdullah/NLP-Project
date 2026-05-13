# Arabic Abstractive Text Summarization using Seq2Seq Models

## Project Overview

This project builds an abstractive Arabic text summarization system using deep learning. The system generates concise summaries from Arabic text while preserving the main ideas of the original article.

Unlike extractive summarization, this approach uses a neural Seq2Seq model to generate new text that captures the meaning of the input.

## Objectives

- Understand and implement a Seq2Seq encoder-decoder architecture
- Apply Bahdanau attention to improve summarization quality
- Preprocess and normalize Arabic text
- Train and evaluate a deep learning summarization model
- Compare generated summaries with reference summaries
- Provide a Streamlit UI for model interaction

## Datasets

The project combines multiple Arabic summarization datasets, including:

- AraSum dataset
- SumArabic dataset
- Egyptian Arabic summarization dataset
- Kaggle Arabic summarization dataset

All datasets are unified into a common structure:

- `text`: input article
- `summarizer`: target summary

## Methodology

### 1. Data Preprocessing

- Load and merge multiple datasets
- Clean Arabic text by removing noise and non-Arabic characters
- Normalize Arabic letters
- Remove null values and duplicates
- Filter short or invalid samples
- Split data into training, validation, and test sets
- Tokenize text and apply sequence padding

### 2. Model Design

The model is based on a Seq2Seq architecture:

- Encoder: Bidirectional LSTM
- Decoder: LSTM
- Attention: Bahdanau attention
- Output layer: softmax over the vocabulary

### 3. Training

- Train the model on the training dataset
- Validate during training
- Use early stopping to reduce overfitting

### 4. Evaluation

The model can be evaluated using:

- ROUGE-1
- ROUGE-2
- ROUGE-L

Qualitative evaluation can also be done by comparing generated summaries with reference summaries.

## Streamlit UI

The project includes a simple Streamlit interface for testing the trained model.

### Run the app locally

Install the required libraries:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Main Files

- `Phase1_Preprocessing.ipynb`: preprocessing phase
- `phase_2_seq2seq.ipynb`: model training phase
- `app.py`: Streamlit UI
- `model_utils.py`: model loading, preprocessing, and summary generation
- `all_outputs/`: saved model, tokenizer, config, and training curve
