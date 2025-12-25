# import necessary libraries
import pandas as pd  # data ingestion
import streamlit as st  # UI
from groq import Groq  # AI interface and Llama model responses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# retrieve api key details from secret file
api_key = st.secrets["API_KEY"]

# Create a groq instance
client = Groq(api_key=api_key)

# Load the data for NLP Intent Recognition
df = pd.read_csv("nlp_intent_dataset.csv")
# Separate x and y features
x = df["User Query"]
y = df["Intent"]

# Create a Pipeline for tfidf vectorizer and naive bayes
pipeline = Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("clf",RandomForestClassifier(n_estimators = 100,max_depth=5))
])

pipeline.fit(x,y)

# Intent response mapping
responses2 = {
    "Password_Reset": "To reset your password, go to settings and click 'Forget Password'.",
    "Check_Balance": "Your current account balance is $5000.",
    "Order_Cancellation": "You can cancel your order from  'My Orders' section.",
    "Order_Status": "Your order is being processed. Check your email for updates."
}


def predict_intent_with_confidence(user_input):
    # Get predicted probabilities for all classes
    probs = pipeline.predict_proba([user_input])[0]
    max_prob = max(probs)
    predicted_intent = pipeline.classes_[probs.argmax()]
    return predicted_intent, max_prob

# Create a function which takes the text ip and checks if the intent is present in the responses file.
# If its present, then return the response operation
# If it is not present, then return the llama model response
def chatbot_response(text):
    intent, confidence = predict_intent_with_confidence(text)

    if intent in responses2 and confidence>0.3:
        return responses2[intent]

    else:
        return llama_response(text)

# Create a function that generates llama model's response
def llama_response(text):
    stream = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role":"system",
                "content":text
            }
        ],
        model = "llama-3.3-70b-versatile",
        stream=True
    )

    for chunk in stream:
        response = chunk.choices[0].delta.content
        if response is not None:
            yield response

# Start building the streamlit app
st.title("Llama Chatbot")
st.subheader("By Salunkhe Ganesh")

text = st.text_area("Please ask any question here : ")

if text:
    st.subheader("Model Response: ")

    # Get response from the chatbot
    response = chatbot_response(text)

    # Handle both static and streamed responses
    if isinstance(response,str):
        st.write(response)   # Predefined response
    else:
        st.write_stream(response)  # Llama response