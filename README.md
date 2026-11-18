# 🧬 Biomedical Abstract NLP Dashboard

An interactive NLP dashboard built with Streamlit for classifying biomedical abstracts and performing advanced text analysis.

## 📌 Features

- **Abstract Classification:** Classifies input text into one of 5 medical categories using a trained **Complement Naive Bayes** model.
- **Prediction Probabilities:** Visualizes confidence scores across all disease classes.
- **Text Statistics & Cleaning:** NLTK tokenization, custom medical stopword removal, and frequency distribution charts.
- **Topic Modeling (LDA):** Analyzes topic distribution with Latent Dirichlet Allocation to identify underlying clinical themes.
- **Semantic Similarity (Word2Vec):** Explores word embeddings and similar terms within biomedical context.
- **Model Evaluation:** Compares benchmark performance (Logistic Regression, SVM, Complement Naive Bayes, Deep Learning).

## 🩺 Disease Categories

1. **Neoplasms:** Cancer-related abstracts, tumors, carcinoma, metastasis, oncology.
2. **Digestive System Diseases:** Colon, liver, gastric, intestinal disorders.
3. **Nervous System Diseases:** Brain, nerve, neurological disorders, syndromes.
4. **Cardiovascular Diseases:** Heart and blood vessel diseases (coronary, myocardial, etc.).
5. **General Pathological Conditions:** Broad disease conditions, infections, complications, and inflammation.

## 📁 Project Structure

```text
├── app.py                            # Main Streamlit application
├── complement_naive_bayes_model.pkl  # Trained Complement Naive Bayes classifier
├── tfidf_vectorizer.pkl              # TF-IDF vectorizer
├── label_encoder.pkl                 # Target label encoder
├── lda_model.pkl                     # Gensim LDA topic model
├── gensim_dictionary.pkl             # Dictionary for LDA
├── word2vec_model.pkl                # Trained Word2Vec model
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <YOUR_REPO_URL>
cd "DS Project"
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
