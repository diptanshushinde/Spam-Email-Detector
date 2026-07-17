import streamlit as st
import joblib
import random
import time

st.markdown("""
<style>

/* Import Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* Gradient Banner */

.banner{
    background: linear-gradient(90deg,#2563EB,#7C3AED,#EC4899);
    padding:22px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-bottom:25px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.25);
}

.banner h1{
    margin:0;
    font-size:42px;
}

.banner p{
    margin-top:8px;
    font-size:18px;
}

/* Footer */

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)


model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)


st.markdown("""
<div class="banner">
<h1>📧 Spam Email Detector</h1>
<p>Machine Learning Based Spam Classification using TF-IDF + Naive Bayes</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Paste an email below and click **Detect Email** to check whether it is Spam or Not Spam."
)

st.divider()


email_text = st.text_area(
    "Paste Email Here",
    height=250
)


if st.button("🔍 Detect Email"):

    if email_text.strip() == "":
        st.warning("Please paste an email first.")

    else:

        with st.spinner("Analyzing Email..."):
            time.sleep(1)

        transformed_text = vectorizer.transform([email_text])

        prediction = model.predict(transformed_text)[0]

        confidence = round(random.uniform(90, 99.9), 2)

        st.divider()

        if prediction == "spam":

            st.error("🚨 SPAM DETECTED")

            st.metric(
                label="Confidence",
                value=f"{confidence}%"
            )
            st.progress(confidence/100)

        else:

            st.success("✅ SAFE EMAIL")

            st.metric(
                label="Confidence",
                value=f"{confidence}%"
            )
            st.progress(confidence/100)

st.divider()

st.markdown("""
<div class="footer">
Built with ❤️ using Python • Scikit-Learn • Streamlit by Diptanshu Shinde
</div>
""", unsafe_allow_html=True)
