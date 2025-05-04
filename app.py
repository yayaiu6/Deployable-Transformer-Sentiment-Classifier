import streamlit as st
from sentiment140_utils import load_model_and_tokenizer, predict



model, tokenizer = load_model_and_tokenizer(
    r"C:\Users\yahya\WORK\Deployable-Transformer-Sentiment-Classifier\pytorch_model.bin",
    r"C:\Users\yahya\WORK\Deployable-Transformer-Sentiment-Classifier"
)


st.title("Sentiment Analysis")



user_input = st.text_input("Enter your sentence:", "")

if st.button("Predict"):
    if user_input:
        prediction = predict(user_input, model, tokenizer)
        if "Negative" in prediction:
            color = "#8B0000"
            emoji = "😑"
        elif "Positive" in prediction:
            color = "green"
            emoji = "😊"
        else:  
            color = "gray"
            emoji = "😐"
        st.markdown(f"<p style='color:{color}; font-weight:bold;'>{prediction} {emoji}</p>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a sentence.")

