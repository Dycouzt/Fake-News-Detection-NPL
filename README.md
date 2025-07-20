# Fake News Detection using RoBERTa

This project uses a fine-tuned RoBERTa transformer model to detect fake news articles. The dataset is sourced from Kaggle’s “Fake and Real News Dataset”. The notebook walks through preprocessing, tokenization, model training, evaluation, and finally provides a simple Streamlit-based web app for real-time inference.

⸻

# Dataset

The dataset consists of news articles labeled as either fake or real, with features including:
	• title: The headline of the news article
	• text: The full body of the news article
	• subject: Topic category (e.g., politics, world news)
	• date: Publication date
	• label: Ground truth classification (1 for fake, 0 for real)

⸻

# Requirements

This project uses the following Python libraries:
	• pandas
	• scikit-learn
	• matplotlib
	• seaborn
	• torch
	• transformers
	• streamlit

Note: To install the required libraries, you can run:

pip install -r requirements.txt


⸻

# Project Structure

FakeNewsDetection/
├── FakeNews_RoBERTa.ipynb     # Jupyter Notebook with full training pipeline
├── app.py                     # Streamlit app for model inference
├── model/                     # Directory containing the saved model/tokenizer
├── requirements.txt           # All dependencies
└── images/
    └── confusion_matrix.png   # Sample evaluation plot


⸻

# Steps
	1. Data Loading: Load the Kaggle Fake and Real News dataset.
	2. Data Cleaning: Remove duplicates, nulls, and unnecessary columns.
	3. Preprocessing:
		• Combine title and text for richer context.
		• Encode labels and shuffle data.
	4. Tokenization:
		• Use Hugging Face’s tokenizer for RoBERTa.
	5. Model Training:
		• Fine-tune RoBERTa on binary classification (fake vs real).
		• Use GPU acceleration if available.
	6. Model Evaluation:
		• Evaluate using accuracy, F1-score, and confusion matrix.
	7. Deployment:
		• Streamlit app (app.py) to load the trained model and allow users to test custom news headlines/text.

⸻

# Code Explanation
	• Jupyter Notebook: Handles all training steps from data exploration to model evaluation.
	• app.py: A simple UI that accepts user input and classifies news as real or fake.
	• model/: Stores the trained model and tokenizer for reuse.

⸻

# Results

The model achieved the following on the test set:
	• Accuracy: 92.5%
	• F1 Score: 0.93
	• Confusion matrix:


⸻

# Future Improvements
	• Add explainability tools like SHAP or LIME to interpret predictions.
	• Expand to multiclass classification with subject prediction.
	• Train on multilingual datasets for broader generalization.
	• Integrate a news feed for live classification.
	• Containerize with Docker for easier deployment.

⸻

# Conclusion

This project demonstrates how transformer models like RoBERTa can be fine-tuned for fake news detection. With a real-world dataset and a live inference app, it showcases the pipeline from raw data to deployable model.

⸻

# Author

Diego Acosta

