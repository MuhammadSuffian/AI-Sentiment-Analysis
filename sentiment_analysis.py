import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
import os
import tempfile
from groq import Groq
import json
import re

# Initialize Groq client with API key from Streamlit secrets
try:
    api_token = st.secrets["api_tokken"]
    client = Groq(api_key=api_token)
except KeyError:
    st.error("API token not found in secrets. Please add 'api_tokken' to your Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Groq client: {str(e)}")
    st.stop()

st.title("🎙️ Voice Sentiment Analyzer")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Audio", "Record Live Audio"])

def clean_response(response):
    # Remove any content between <think> tags
    cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    # Remove any remaining <think> tags
    cleaned_response = cleaned_response.replace('<think>', '').replace('</think>', '')
    # Remove any extra whitespace
    cleaned_response = ' '.join(cleaned_response.split())
    return cleaned_response

def analyze_sentiment_and_get_llm_response(text):
    # Sentiment analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    
    # Generate LLM response based on sentiment
    prompt = f"""
    Based on the following text and its sentiment ({sentiment_label}, score: {sentiment:.2f}), 
    provide a thoughtful and empathetic response. If the sentiment is positive, acknowledge and reinforce the positive aspects. 
    If negative, offer support and constructive suggestions. If neutral, provide balanced insights.
    
    Text: {text}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and empathetic assistant that responds to user input based on their emotional state."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=1024,
        )
        
        llm_response = chat_completion.choices[0].message.content
        # Clean the response to remove any think tags
        cleaned_response = clean_response(llm_response)
        return sentiment, sentiment_label, cleaned_response
    except Exception as e:
        st.error(f"Error getting LLM response: {str(e)}")
        return sentiment, sentiment_label, None

with tab1:
    uploaded_file = st.file_uploader("Upload an audio file (WAV/MP3)", type=["wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file)

        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile("temp_audio.wav")

        with audio_file as source:
            st.info("Converting speech to text...")
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                st.success("Transcription:")
                st.write(text)

                # Get sentiment analysis and LLM response
                sentiment, sentiment_label, llm_response = analyze_sentiment_and_get_llm_response(text)
                
                st.subheader("Sentiment Analysis Result")
                st.metric(label="Sentiment", value=sentiment_label, delta=f"{sentiment:.2f}")
                
                if llm_response:
                    st.subheader("AI Response")
                    st.write(llm_response)

            except sr.UnknownValueError:
                st.error("Could not understand audio.")
            except sr.RequestError:
                st.error("Could not request results from the speech recognition service.")

with tab2:
    st.subheader("Record Audio Live")
    st.write("Click the button below to record audio from your microphone:")
    
    # Using Streamlit's audio input component
    audio_bytes = st.audio_input("Click to record")
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # Save the recorded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            # Read the content before writing - this fixes the bytes-like object issue
            audio_content = audio_bytes.read()
            tmp_file.write(audio_content)
            temp_filename = tmp_file.name
        
        try:
            # Process the recorded audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_filename) as source:
                st.info("Converting speech to text...")
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    st.success("Transcription:")
                    st.write(text)

                    # Get sentiment analysis and LLM response
                    sentiment, sentiment_label, llm_response = analyze_sentiment_and_get_llm_response(text)
                    
                    st.subheader("Sentiment Analysis Result")
                    st.metric(label="Sentiment", value=sentiment_label, delta=f"{sentiment:.2f}")
                    
                    if llm_response:
                        st.subheader("AI Response")
                        st.write(llm_response)

                except sr.UnknownValueError:
                    st.error("Could not understand audio.")
                except sr.RequestError:
                    st.error("Could not request results from the speech recognition service.")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
