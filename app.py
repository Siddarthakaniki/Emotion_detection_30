# app.py
# Text Emotion Detection - Streamlit app with spaCy preprocessing
# Paste this file into your repo root (next to model/ and data/), then deploy.

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import joblib
import re

# ---------------------
# Configuration / Paths
# ---------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "text_emotion.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "emotion_dataset_raw.csv")

# ---------------------
# Helper: Load spaCy (optional)
# ---------------------
nlp = None
spacy_available = False
try:
    import spacy
    # Attempt to load the small English model
    try:
        nlp = spacy.load("en_core_web_sm")
        spacy_available = True
    except Exception:
        # spaCy installed but model not present
        spacy_available = False
except Exception:
    spacy_available = False

# ---------------------
# Text preprocessing
# ---------------------
def preprocess_with_spacy(text):
    """
    Use spaCy to lemmatize, remove stopwords, punctuations, non-alpha tokens.
    Returns a cleaned string.
    """
    if not spacy_available or nlp is None:
        return simple_preprocess(text)

    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if not token.is_alpha:
            continue
        lemma = token.lemma_.strip().lower()
        if lemma:
            tokens.append(lemma)
    return " ".join(tokens)


def simple_preprocess(text):
    """Fallback simple cleaning: lowercase, remove non-alpha, collapse spaces."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove urls
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text):
    """Main preprocessing wrapper: prefer spaCy if available."""
    if spacy_available and nlp is not None:
        return preprocess_with_spacy(text)
    else:
        return simple_preprocess(text)

# ---------------------
# Load ML model safely
# ---------------------
pipe_lr = None
if os.path.exists(MODEL_PATH):
    try:
        pipe_lr = joblib.load(open(MODEL_PATH, "rb"))
    except Exception as e:
        # model exists but failed to load
        pipe_lr = None
        # Delay st.error() until streamlit context (below)
else:
    pipe_lr = None

# ---------------------
# Emoji dictionary
# ---------------------
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# ---------------------
# Prediction helpers
# ---------------------
def predict_emotions(text):
    if pipe_lr is None:
        return "Model_Not_Loaded"
    cleaned = preprocess(text)
    try:
        return pipe_lr.predict([cleaned])[0]
    except Exception:
        # fallback: try raw text if model expects raw text
        try:
            return pipe_lr.predict([text])[0]
        except Exception:
            return "Prediction_Error"

def get_prediction_proba(text):
    if pipe_lr is None:
        return np.array([[0.0]])
    cleaned = preprocess(text)
    try:
        return pipe_lr.predict_proba([cleaned])
    except Exception:
        try:
            return pipe_lr.predict_proba([text])
        except Exception:
            return np.array([[0.0]])

# ---------------------
# Example test sentences (covers many emotions)
# ---------------------
EXAMPLES = {
    "Anger": "I can't believe this — I'm so angry and frustrated!",
    "Disgust": "That was disgusting, I can't even look at it.",
    "Fear": "I'm terrified about the result tomorrow.",
    "Happy": "I am so happy to see you! Best day ever!",
    "Joy": "LOL that joke made me laugh so hard!",
    "Neutral": "I went to the store to buy milk and eggs.",
    "Sad": "I feel really down and sad today.",
    "Shame": "I am embarrassed; I shouldn't have said that.",
    "Surprise": "Wow, I didn't expect that at all!"
}

# ---------------------
# Streamlit UI
# ---------------------
def main():
    st.set_page_config(page_title="Text Emotion Detection", page_icon="💬", layout="wide")
    st.title("💬 Text Emotion Detection (with spaCy preprocessing)")
    st.markdown(
        "Enter text and the model will predict the emotion. "
        "Preprocessing uses **spaCy** if available; otherwise a fallback cleaner is used."
    )

    # Show spaCy/model status
    col_status, col_model = st.columns([1, 2])
    with col_status:
        if spacy_available:
            st.success("spaCy: available")
        else:
            st.warning("spaCy: not available — using simple preprocessing")
    with col_model:
        if pipe_lr is None:
            st.error("Model: not loaded. Ensure `model/text_emotion.pkl` exists and is a compatible model.")
        else:
            st.success("Model: loaded")

    # Sidebar - menu
    menu = ["Home", "Examples", "Explore Dataset", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Type text for emotion detection")
        with st.form(key="predict_form"):
            user_text = st.text_area("Enter text here...", height=140)
            submitted = st.form_submit_button("Predict")
        if submitted:
            if not user_text or str(user_text).strip() == "":
                st.info("Please enter some text to analyze.")
            else:
                prediction = predict_emotions(user_text)
                prob = get_prediction_proba(user_text)

                # left: text + prediction, right: probabilities chart
                left, right = st.columns([2, 1])
                with left:
                    st.markdown("**Original text:**")
                    st.write(user_text)

                    st.markdown("**Prediction:**")
                    emoji = emotions_emoji_dict.get(prediction, "")
                    st.write(f"**{prediction}** {emoji}")

                    try:
                        conf = np.max(prob)
                        st.write(f"**Confidence:** {conf:.2f}")
                    except Exception:
                        st.write("Confidence: N/A")

                with right:
                    st.markdown("**Probability distribution**")
                    # build dataframe for chart
                    if pipe_lr is not None:
                        try:
                            classes = list(pipe_lr.classes_)
                            proba_df = pd.DataFrame(prob, columns=classes)
                            proba_df_clean = proba_df.T.reset_index()
                            proba_df_clean.columns = ["Emotion", "Probability"]
                        except Exception:
                            proba_df_clean = pd.DataFrame([{"Emotion": "N/A", "Probability": 0.0}])
                    else:
                        proba_df_clean = pd.DataFrame([{"Emotion": "N/A", "Probability": 0.0}])
                    chart = alt.Chart(proba_df_clean).mark_bar().encode(
                        x="Emotion",
                        y="Probability",
                        color="Emotion"
                    )
                    st.altair_chart(chart, use_container_width=True)

    elif choice == "Examples":
        st.subheader("Quick examples (click a button to test)")
        cols = st.columns(3)
        items = list(EXAMPLES.items())
        for i, (label, sentence) in enumerate(items):
            col = cols[i % 3]
            if col.button(label):
                prediction = predict_emotions(sentence)
                prob = get_prediction_proba(sentence)
                st.write(f"**Example ({label})**: {sentence}")
                st.write(f"**Predicted:** {prediction} {emotions_emoji_dict.get(prediction, '')}")
                try:
                    st.write(f"**Confidence:** {np.max(prob):.2f}")
                except Exception:
                    st.write("Confidence: N/A")
                # show probabilities
                if pipe_lr is not None:
                    classes = list(pipe_lr.classes_)
                    proba_df = pd.DataFrame(prob, columns=classes)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["Emotion", "Probability"]
                    st.altair_chart(alt.Chart(proba_df_clean).mark_bar().encode(x='Emotion', y='Probability', color='Emotion'), use_container_width=True)

    elif choice == "Explore Dataset":
        st.subheader("Dataset preview")
        if os.path.exists(DATA_PATH):
            try:
                df = pd.read_csv(DATA_PATH)
                st.write(df.head(10))
                st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
        else:
            st.error("Dataset not found at `data/emotion_dataset_raw.csv`. Upload it into the repo's data/ folder.")

    else:  # About
        st.subheader("About this project")
        st.markdown("""
        **Text Emotion Detection** using classic ML and NLP preprocessing.
        - Preprocessing: spaCy (lemmatization, stopword removal) if available; otherwise a simple fallback.
        - Model: pre-trained model loaded from `model/text_emotion.pkl` (saved with joblib).
        - UI: Streamlit with probability visualization (Altair).
        """)

        st.markdown("**Notes & tips**")
        st.markdown("- If spaCy is not installed on the server, add `spacy` and the `en_core_web_sm` model to `requirements.txt`.")
        st.markdown("- If your model file is large (>100 MB), host it externally or use Git LFS.")

if __name__ == "__main__":
    main()
