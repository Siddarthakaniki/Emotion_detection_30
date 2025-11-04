import os

# Robust model path resolution: find model relative to this file
BASE_DIR = os.path.dirname(__file__)
MODEL_RELATIVE = os.path.join(BASE_DIR, "model", "text_emotion.pkl")
if not os.path.exists(MODEL_RELATIVE):
    # try one level up (in case app.py is inside a subfolder)
    MODEL_RELATIVE = os.path.join(BASE_DIR, "..", "model", "text_emotion.pkl")
    MODEL_RELATIVE = os.path.normpath(MODEL_RELATIVE)

if not os.path.exists(MODEL_RELATIVE):
    # Provide a clear Streamlit error at runtime (prevents crash during import)
    try:
        import streamlit as st
        st.error(f"Model file not found at: {MODEL_RELATIVE}. Please upload 'text_emotion.pkl' in the 'model' folder of the repo.")
    except Exception:
        pass
    # fallback: do not raise here to avoid import-time crash; set pipe_lr to None
    pipe_lr = None
else:
    )

import streamlit as st

import pandas as pd
import numpy as np
import altair as alt


import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "text_emotion.pkl")

if os.path.exists(MODEL_PATH):
    try:
        pipe_lr = joblib.load(open(MODEL_PATH, "rb"))
    except Exception as e:
        pipe_lr = None
        st.error(f"Error loading model: {e}")
else:
    pipe_lr = None
    st.error("Model file not found at 'model/text_emotion.pkl'. Please upload it to the repository's model/ folder.")

)

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx]) if pipe_lr is not None else ['Model_Not_Loaded'] if pipe_lr is not None else ['model_not_loaded']
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx]) if pipe_lr is not None else [[0.0]] if pipe_lr is not None else [[0.0]]
    return results


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)






if __name__ == '__main__':
    main()
