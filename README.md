# 🧬 Automated Biomedical Text Classification & Topic Discovery Platform

An interactive NLP dashboard and end-to-end machine learning pipeline for classifying biomedical abstracts and performing advanced text analysis.

---

## 📌 Features

- **Abstract Classification:** Classifies input biomedical abstracts into 5 disease categories using a trained **Complement Naive Bayes** classifier.
- **Prediction Probabilities:** Visualizes confidence scores across all disease classes.
- **Text Statistics & Cleaning:** NLTK tokenization, custom medical stopword removal, and term frequency distribution charts.
- **Topic Modeling (LDA):** Unsupervised Latent Dirichlet Allocation (LDA) to extract underlying clinical themes.
- **Semantic Similarity (Word2Vec):** Explores word embeddings and semantically similar medical terms.
- **Model Evaluation:** Benchmarks and compares multiple ML and Deep Learning architectures.

---

## 🩺 Disease Categories

1. **Neoplasms:** Cancer-related abstracts, tumors, carcinoma, metastasis, oncology.
2. **Digestive System Diseases:** Colon, liver, gastric, and intestinal disorders.
3. **Nervous System Diseases:** Brain, nerve, neurological disorders, syndromes, and nervous system conditions.
4. **Cardiovascular Diseases:** Heart and blood vessel diseases (coronary, myocardial, arterial, etc.).
5. **General Pathological Conditions:** Broad disease conditions, infections, complications, and systemic inflammation.

---

## 📊 Model Performance Comparison

The models were evaluated on TF-IDF representations of the abstracts using Accuracy and Macro F1 score:

| Model | Accuracy | Macro F1 | Status |
| :--- | :---: | :---: | :--- |
| **Complement Naive Bayes** | **59.59%** | **58.77%** | 🏆 **Best Model (Deployed in App)** |
| **Logistic Regression** | 57.86% | 57.65% | Strong Linear Baseline |
| **Deep Learning (Feedforward NN)** | 57.41% | 56.53% | Dense Layers + Dropout |
| **Linear SVM** | 54.22% | 53.93% | Baseline Classifier |

*Note: Complement Naive Bayes performed best due to its suitability for imbalanced, sparse TF-IDF text features in multi-class biomedical classification.*

---

## 🔬 Discovered Clinical Topics (LDA)

| Topic | Clinical Interpretation | Representative Terms |
| :--- | :--- | :--- |
| **Topic 1** | Cellular pathology & biochemical analysis | `cell`, `level`, `serum`, `plasma`, `tumor` |
| **Topic 2** | Cardiovascular diseases | `coronary`, `ventricular`, `artery`, `myocardial` |
| **Topic 3** | General pathological & clinical conditions | `disease`, `risk`, `blood`, `pressure`, `age` |
| **Topic 4** | Surgical & postoperative complications | `surgery`, `complication`, `postoperative`, `operation` |
| **Topic 5** | Neoplasms & cancer diagnosis | `tumor`, `cancer`, `carcinoma`, `diagnosis`, `lesion` |

---

## 📁 Project Structure

```text
├── app.py                            # Streamlit interactive web application
├── Project.py                        # Full data science training & NLP pipeline
├── Report.pdf                        # Detailed academic project report & analysis
├── complement_naive_bayes_model.pkl  # Trained Complement Naive Bayes classifier
├── tfidf_vectorizer.pkl              # Fitted TF-IDF vectorizer
├── label_encoder.pkl                 # Target label encoder
├── lda_model.pkl                     # Gensim LDA topic model
├── gensim_dictionary.pkl             # Dictionary for LDA topic model
├── word2vec_model.pkl                # Trained Word2Vec word embeddings
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mdomarabd/Biomedical-Text-Classification.git
cd Biomedical-Text-Classification
```

### 2. Create and Activate a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Launch the interactive web dashboard locally:
```bash
streamlit run app.py
```

### 5. (Optional) Run the Training Pipeline
To retrain models from scratch and regenerate model artifacts:
```bash
python Project.py
```
*(Note: requires train and test CSV datasets).*

---

## 📄 Documentation

For an in-depth explanation of the methodology, exploratory data analysis, feature engineering, and model evaluation, please refer to [Report.pdf](Report.pdf).
