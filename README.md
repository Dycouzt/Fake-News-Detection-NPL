# Fake News Detection Using RoBERTa and NLP

This project fine-tunes a RoBERTa transformer model to classify news articles as real or fake using the Fake-and-Real-News dataset from Kaggle. It applies Natural Language Processing (NLP) techniques to clean, tokenize, and vectorize text before feeding it to a binary classification model. A lightweight Streamlit app is included for real-time inference.

---

## Goal

The goal of this project is to build an end-to-end pipeline for detecting misinformation using a pre-trained transformer model. It explores how deep NLP models like RoBERTa can be applied to real-world fake news datasets, and provides a simple interface for testing predictions interactively.

---

## Features

- Loads and cleans labeled news data from Kaggle
- Combines headlines and article body for richer input context
- Fine-tunes a RoBERTa transformer on binary classification (real vs fake)
- Evaluates with accuracy, F1 score, and confusion matrix
- Saves the model and tokenizer for reuse
- Deploys a simple inference tool via Streamlit (`app.py`)
- Enables user-input predictions in real time

---

## Requirements

This project uses the following Python libraries:
- pandas (for data processing)
- scikit-learn (for metrics and preprocessing)
- torch (PyTorch backend for model training)
- transformers (Hugging Face RoBERTa implementation)
- matplotlib and seaborn (for evaluation plots)
- streamlit (for model testing UI)

Install the required libraries with:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```plaintext
  FakeNewsDetection/
├── fakenewsprime.ipynb
├── app.py
└── README.md 
```      

---

## Dataset

fake-and-real-news-dataset.csv

---

## How It Works

1. Data Loading:
   - Loads the "Fake and Real News" dataset from Kaggle.
   - Combines title and text columns for better semantic context.

2. Preprocessing:
   - Removes null values and unnecessary columns.
   - Shuffles and encodes labels (`0` for real, `1` for fake).

3. Tokenization:
   - Uses RoBERTa tokenizer from Hugging Face Transformers.
   - Converts news text to input IDs and attention masks.

4. Model Training:
   - Fine-tunes RoBERTa on the dataset using binary cross-entropy loss.
   - Utilizes GPU if available for faster convergence.

5. Evaluation:
   - Computes accuracy, F1-score, and plots a confusion matrix.
   - Saves trained model and tokenizer.

6. Deployment:
   - app.py loads the trained model and runs a Streamlit interface.
   - Users can input news headlines or full articles to get predictions.

---

## Code Highlights

- `transformers.RobertaTokenizer` and `RobertaForSequenceClassification`: Used for tokenization and model architecture.
- `Trainer` and `TrainingArguments`: Simplify fine-tuning setup.
- `sklearn.metrics.f1_score`: Evaluates model on test data.
- `streamlit.text_input()` + `model.predict()`: Enables real-time testing.

---

## Sample Output

```plaintext
Training complete.
Validation Accuracy: 92.5%
F1 Score: 0.93

Enter news headline or article:
> The government launches new economic reform bill.

Prediction: REAL NEWS ✅
```

## To tun the Streamlit app:

```bash
streamlit run app.py
```

You have to make sure the model and the tokenizer are in the correct model / directory.

---

## Conclusion

This project demonstrates how transformer-based NLP models can be fine-tuned for misinformation detection. With a clean dataset, effective preprocessing, and powerful language modeling, RoBERTa achieves strong performance on binary classification. The Streamlit interface provides a quick way to test and visualize results, making the model useful for education, journalism, or security research.

---

## Author

Diego Acosta - Dycouzt
