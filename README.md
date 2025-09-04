# Auto-Complete-System-using-N-gram-Language-Models
Implementation of an Auto-Complete System using N-gram Language Models with text preprocessing, probability estimation, perplexity evaluation, and word prediction.
# 🔍 Auto-Complete System using N-gram Language Models

This project implements an **Auto-Complete System** using **N-gram Language Models** from scratch in Python.  
The system predicts the most probable next word in a sentence based on learned probabilities from a large text dataset.

---

## 🚀 Project Overview

An **auto-complete system** predicts the next possible word or phrase given a partial input.  
For example:

> **Input:**  
`I love natural`

> **Prediction:**  
`I love natural language`

In this project, we use an **N-gram based statistical language model** that learns from a training dataset and predicts the **most likely** next word based on conditional probabilities.

---

## 🧠 Key Concepts Implemented

### **1. Text Preprocessing**
- Lowercasing, punctuation removal, and tokenization.
- Replacing low-frequency words with `<unk>` tokens.
- Splitting the dataset into training and testing sets.

### **2. N-gram Language Model**
- **Unigram**, **Bigram**, **Trigram**, and higher-order N-grams supported.
- Calculates conditional probabilities using **maximum likelihood estimation**.
- Implements **Laplace (k) smoothing** to handle unseen words.

### **3. Perplexity Evaluation**
Perplexity measures how well the language model predicts the test data.  
Lower perplexity → Better predictions.

### **4. Auto-Complete Predictions**
Given a partial sentence, the model predicts the **top N most probable next words**.

---

## 📌 Implementation Details

### **1. Preprocess and Tokenize Sentences**
```python
def tokenize_sentences(text):
    sentences = text.lower().strip().split("\n")
    return [sentence.split() for sentence in sentences]
