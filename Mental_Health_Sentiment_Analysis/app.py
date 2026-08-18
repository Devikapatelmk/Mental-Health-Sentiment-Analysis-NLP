import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    page_icon="🧠",
    layout="centered"
)

# ============================================================
# NLTK RESOURCE SETUP
# ============================================================

def setup_nltk():

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        WordNetLemmatizer().lemmatize("tests")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


setup_nltk()

# ============================================================
# STOPWORDS AND LEMMATIZER
# ============================================================

stop_words = set(stopwords.words("english"))

# Keep important negation words
important_negations = {
    "no",
    "not",
    "nor",
    "never"
}

stop_words = stop_words - important_negations

lemmatizer = WordNetLemmatizer()

# ============================================================
# TEXT PREPROCESSING FUNCTION
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+\.\S+",
        " ",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Keep hashtag text
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # CONTRACTION HANDLING
    # --------------------------------------------------------

    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bn't\b", " not", text)

    text = re.sub(r"\bi'm\b", "i am", text)
    text = re.sub(r"\bit's\b", "it is", text)
    text = re.sub(r"\bi've\b", "i have", text)
    text = re.sub(r"\bi'll\b", "i will", text)
    text = re.sub(r"\bi'd\b", "i would", text)

    text = re.sub(r"\byou're\b", "you are", text)
    text = re.sub(r"\bthey're\b", "they are", text)
    text = re.sub(r"\bwe're\b", "we are", text)

    text = re.sub(r"\bdoesn't\b", "does not", text)
    text = re.sub(r"\bdon't\b", "do not", text)
    text = re.sub(r"\bdidn't\b", "did not", text)
    text = re.sub(r"\bisn't\b", "is not", text)
    text = re.sub(r"\baren't\b", "are not", text)
    text = re.sub(r"\bwasn't\b", "was not", text)
    text = re.sub(r"\bweren't\b", "were not", text)

    # Remove numbers, punctuation and special characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
        and len(word) > 1
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

# ============================================================
# MODEL FILE PATHS
# ============================================================

MODEL_FOLDER = "mental_health_model_files"

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "mental_health_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_FOLDER,
    "tfidf_vectorizer.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_FOLDER,
    "label_encoder.pkl"
)

# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    return model, vectorizer, label_encoder


try:

    model, vectorizer, label_encoder = load_model_files()

except Exception as error:

    st.error(
        "Model files could not be loaded."
    )

    st.write(
        "Make sure your folder structure is:"
    )

    st.code(
        """
Mental_Health_Sentiment_Analysis
│
├── app.py
│
└── mental_health_model_files
    ├── mental_health_model.pkl
    ├── tfidf_vectorizer.pkl
    └── label_encoder.pkl
        """
    )

    st.exception(error)

    st.stop()

# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🧠 Mental Health Sentiment Analysis"
)

st.markdown(
    """
    This NLP application analyzes a text statement and predicts
    the most likely **mental-health-related text category**.

    The model was trained using **TF-IDF + Logistic Regression**.
    """
)

# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ Educational project only. "
    "This system does not provide medical or psychological diagnosis."
)

st.divider()

# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "Enter a Statement"
)

user_text = st.text_area(
    "Type your text below:",
    placeholder=(
        "Example: I have been feeling nervous and "
        "worried lately and I cannot relax."
    ),
    height=170
)

# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Text",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if analyze_button:

    if not user_text.strip():

        st.warning(
            "Please enter some text before clicking Analyze."
        )

    else:

        cleaned_text = clean_text(
            user_text
        )

        if cleaned_text.strip() == "":

            st.warning(
                "The entered statement became empty "
                "after preprocessing. Please enter more meaningful text."
            )

        else:

            # ------------------------------------------------
            # TF-IDF TRANSFORMATION
            # ------------------------------------------------

            text_vector = vectorizer.transform(
                [cleaned_text]
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            predicted_label = model.predict(
                text_vector
            )[0]

            predicted_category = (
                label_encoder.inverse_transform(
                    [predicted_label]
                )[0]
            )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            probabilities = model.predict_proba(
                text_vector
            )[0]

            confidence = (
                probabilities[
                    predicted_label
                ]
                * 100
            )

            # =================================================
            # RESULT SECTION
            # =================================================

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            st.success(
                f"Predicted Category: **{predicted_category}**"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Prediction",
                    predicted_category
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            # =================================================
            # CONFIDENCE MESSAGE
            # =================================================

            if confidence >= 75:

                st.success(
                    "High model confidence."
                )

            elif confidence >= 50:

                st.info(
                    "Moderate model confidence."
                )

            else:

                st.warning(
                    "Low model confidence. "
                    "The statement may be ambiguous."
                )

            # =================================================
            # CLEANED TEXT
            # =================================================

            with st.expander(
                "View Preprocessed Text"
            ):

                st.write(
                    cleaned_text
                )

            # =================================================
            # PROBABILITY TABLE
            # =================================================

            probability_df = pd.DataFrame({
                "Category":
                    label_encoder.classes_,

                "Probability (%)":
                    probabilities * 100
            })

            probability_df = (
                probability_df
                .sort_values(
                    by="Probability (%)",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            probability_df[
                "Probability (%)"
            ] = (
                probability_df[
                    "Probability (%)"
                ]
                .round(2)
            )

            st.subheader(
                "Category Probability Distribution"
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # BAR CHART
            # =================================================

            st.subheader(
                "Probability Chart"
            )

            chart_df = probability_df.copy()

            chart_df = chart_df.set_index(
                "Category"
            )

            st.bar_chart(
                chart_df[
                    "Probability (%)"
                ]
            )

# ============================================================
# SAMPLE INPUTS
# ============================================================

st.divider()

st.subheader(
    "Sample Statements"
)

st.write(
    """
    You can try statements such as:

    - I feel nervous and worried all the time and I cannot relax.
    - I feel extremely sad and hopeless lately.
    - I am feeling calm and happy today.
    - I feel stressed because I have too much work.
    - I keep thinking about ending my life.
    """
)

# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "Model Information"
)

model_info = pd.DataFrame({

    "Property": [
        "Algorithm",
        "Feature Extraction",
        "TF-IDF Features",
        "Number of Classes",
        "Test Accuracy",
        "Weighted F1 Score",
        "Macro F1 Score"
    ],

    "Value": [
        "Logistic Regression",
        "TF-IDF",
        "30,000",
        "7",
        "74.12%",
        "0.7407",
        "0.6854"
    ]

})

st.dataframe(
    model_info,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# CLASS INFORMATION
# ============================================================

st.subheader(
    "Prediction Categories"
)

st.write(
    """
    The model can predict the following categories:

    - Anxiety
    - Bipolar
    - Depression
    - Normal
    - Personality disorder
    - Stress
    - Suicidal
    """
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sentiment Analysis for Mental Health Monitoring | "
    "NLP Major Project | Data Science MITM"
)