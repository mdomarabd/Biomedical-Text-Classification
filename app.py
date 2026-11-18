import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# ==============================
# LOAD MODELS
# ==============================

model = joblib.load("complement_naive_bayes_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
lda_model = joblib.load("lda_model.pkl")
dictionary = joblib.load("gensim_dictionary.pkl")
word2vec_model = joblib.load("word2vec_model.pkl")

label_names = {
    1: "Neoplasms",
    2: "Digestive System Diseases",
    3: "Nervous System Diseases",
    4: "Cardiovascular Diseases",
    5: "General Pathological Conditions"
}

label_descriptions = {
    1: "Cancer-related abstracts, tumors, carcinoma, metastasis, oncology.",
    2: "Digestive system diseases such as colon, liver, gastric, intestinal disorders.",
    3: "Brain, nerve, neurological disorders, syndromes, and nervous system conditions.",
    4: "Heart and blood vessel diseases such as coronary, artery, ventricular, myocardial disorders.",
    5: "Broad pathological conditions, infections, complications, inflammation, and general disease states."
}

# ==============================
# CLEANING FUNCTION
# ==============================

stop_words = set(stopwords.words("english"))

academic_stopwords = {
    "study", "studies", "patient", "patients", "method", "methods",
    "results", "result", "conclusion", "conclusions", "associated",
    "compared", "group", "groups", "using", "used", "showed", "found",
    "performed", "cases", "case", "clinical", "effect", "effects",
    "treatment", "treated", "one", "two", "three", "four", "five",
    "may", "less", "greater", "mean", "significant", "significantly",
    "within", "without", "among", "either", "well", "observed",
    "report", "reported", "increase", "increased"
}

stop_words = stop_words.union(academic_stopwords)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(
    page_title="Biomedical Abstract NLP Dashboard",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Biomedical Abstract NLP Dashboard")

st.write(
    "This application classifies biomedical abstracts and demonstrates NLP techniques "
    "including NLTK preprocessing, TF-IDF classification, LDA topic modeling, and Word2Vec semantic similarity."
)

# Sidebar
st.sidebar.title("Project Information")

st.sidebar.write("**Dataset:** Medical Text Classification Dataset")
st.sidebar.write("**Source:** Kaggle")
st.sidebar.write("**Best Model:** Complement Naive Bayes")
st.sidebar.write("**Features:** TF-IDF")
st.sidebar.write("**Categories:** 5 biomedical classes")

st.sidebar.subheader("Class Labels")

for label, name in label_names.items():
    st.sidebar.write(f"**{label}:** {name}")

# Main input
st.subheader("Enter a Biomedical Abstract")

user_text = st.text_area(
    "Paste a biomedical abstract here:",
    height=250
)

if st.button("Analyze Abstract"):

    if user_text.strip() == "":
        st.warning("Please enter a biomedical abstract.")

    else:
        cleaned = clean_text(user_text)
        tokens = cleaned.split()

        vectorized = tfidf.transform([cleaned])

        prediction = model.predict(vectorized)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        predicted_category = label_names[predicted_label]

        st.success("Analysis complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Result")
            st.metric(
                label="Predicted Category",
                value=f"{predicted_label}: {predicted_category}"
            )

            st.write("**Category Description:**")
            st.info(label_descriptions[predicted_label])

        with col2:
            st.subheader("Text Statistics")
            st.write(f"**Original word count:** {len(user_text.split())}")
            st.write(f"**Cleaned word count:** {len(tokens)}")
            st.write(f"**Unique cleaned terms:** {len(set(tokens))}")

        # ==============================
        # Prediction probabilities
        # ==============================

        if hasattr(model, "predict_proba"):

            st.subheader("Prediction Probability by Category")

            probabilities = model.predict_proba(vectorized)[0]

            prob_df = pd.DataFrame({
                "Category": [
                    f"{label}: {label_names[label]}"
                    for label in label_encoder.classes_
                ],
                "Probability": probabilities
            })

            st.bar_chart(
                prob_df.set_index("Category")
            )

            st.dataframe(prob_df)

        # ==============================
        # Cleaned text
        # ==============================

        st.subheader("Cleaned Text Preview")

        st.write(cleaned[:1500])

        # ==============================
        # Frequent terms
        # ==============================

        st.subheader("Most Frequent Terms in This Abstract")

        if len(tokens) > 0:
            term_counts = Counter(tokens).most_common(15)

            term_df = pd.DataFrame(
                term_counts,
                columns=["Term", "Frequency"]
            )

            st.bar_chart(term_df.set_index("Term"))
            st.dataframe(term_df)

        # ==============================
        # LDA topic distribution
        # ==============================

        st.subheader("LDA Topic Distribution")

        bow = dictionary.doc2bow(tokens)

        topic_distribution = lda_model.get_document_topics(
            bow,
            minimum_probability=0
        )

        topic_labels = {
            0: "Cellular / Biochemical Analysis",
            1: "Cardiovascular Diseases",
            2: "General Clinical Risk Factors",
            3: "Surgery / Postoperative Complications",
            4: "Neoplasms / Cancer Diagnosis"
        }

        topic_df = pd.DataFrame({
            "Topic": [
                topic_labels.get(topic_id, f"Topic {topic_id}")
                for topic_id, prob in topic_distribution
            ],
            "Probability": [
                prob for topic_id, prob in topic_distribution
            ]
        })

        st.bar_chart(topic_df.set_index("Topic"))
        st.dataframe(topic_df)

        dominant_topic = topic_df.sort_values(
            by="Probability",
            ascending=False
        ).iloc[0]

        st.info(
            f"Dominant topic: **{dominant_topic['Topic']}** "
            f"with probability {dominant_topic['Probability']:.2f}"
        )

        # ==============================
        # Word2Vec similarity
        # ==============================

        st.subheader("Word2Vec Semantic Similarity")

        selected_word = st.selectbox(
            "Choose a word from the abstract:",
            options=tokens if len(tokens) > 0 else ["No tokens available"]
        )

        if selected_word in word2vec_model.wv:

            similar_words = word2vec_model.wv.most_similar(
                selected_word,
                topn=10
            )

            sim_df = pd.DataFrame(
                similar_words,
                columns=["Similar Word", "Similarity"]
            )

            st.write(f"Words semantically similar to **{selected_word}**:")
            st.dataframe(sim_df)

            st.bar_chart(sim_df.set_index("Similar Word"))

        else:
            st.warning(
                "This word was not found in the Word2Vec vocabulary."
            )

# ==============================
# Static project sections
# ==============================

st.divider()

st.subheader("Model Performance Summary")

performance_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "SVM",
        "Complement Naive Bayes",
        "Deep Learning"
    ],
    "Accuracy": [
        0.578601,
        0.542244,
        0.595914,
        0.574100
    ],
    "Macro F1": [
        0.576504,
        0.539311,
        0.587697,
        0.565326
    ]
})

st.dataframe(performance_df)
st.bar_chart(performance_df.set_index("Model"))

st.subheader("LDA Topics Discovered")

topics_table = pd.DataFrame({
    "Topic": [
        "Topic 1",
        "Topic 2",
        "Topic 3",
        "Topic 4",
        "Topic 5"
    ],
    "Interpretation": [
        "Cellular pathology and biochemical analysis",
        "Cardiovascular diseases",
        "General pathological and clinical conditions",
        "Surgical and postoperative complications",
        "Neoplasms and cancer diagnosis"
    ],
    "Representative Terms": [
        "cell, level, serum, plasma, tumor",
        "coronary, ventricular, artery, myocardial",
        "disease, risk, blood, pressure, age",
        "surgery, complication, postoperative, operation",
        "tumor, cancer, carcinoma, diagnosis, lesion"
    ]
})

st.dataframe(topics_table)

st.subheader("Project Conclusion")

st.write(
    """
    Complement Naive Bayes achieved the best performance because biomedical abstracts
    are represented as sparse TF-IDF vectors. The app also shows that some categories
    overlap strongly, especially General Pathological Conditions, which shares medical
    vocabulary with cancer, cardiovascular, digestive, and nervous system abstracts.
    """
)