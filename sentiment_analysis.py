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


def format_llm_response(text: str) -> str:
    """Format LLM output for Streamlit display:
    - Preserve fenced code blocks unchanged
    - Preserve markdown tables as contiguous blocks and ensure a blank line before/after a table
    - Preserve list items (no blank lines inserted between list items)
    - For normal lines, insert a blank line after single newlines so paragraphs are separated
    """
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Ensure any inline '###' headings are moved to their own line
    # If '###' is not already at the start of a line, insert a newline before it
    text = re.sub(r'(?<!\n)(?=###)', '\n', text)

    lines = text.split('\n')
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fenced code block: copy until closing fence unchanged
        if stripped.startswith('```'):
            if out_lines and out_lines[-1] != '':
                out_lines.append('')
            out_lines.append(line)
            i += 1
            while i < len(lines):
                out_lines.append(lines[i])
                if lines[i].strip().startswith('```'):
                    i += 1
                    break
                i += 1
            if out_lines and out_lines[-1] != '':
                out_lines.append('')
            continue

        # Markdown table detection: contiguous lines containing a pipe '|' are treated as a table block
        if '|' in line and not stripped.startswith(('-', '*', '+')):
            # Collect contiguous pipe lines
            if out_lines and out_lines[-1] != '':
                out_lines.append('')
            j = i
            while j < len(lines) and '|' in lines[j]:
                out_lines.append(lines[j])
                j += 1
            if out_lines and out_lines[-1] != '':
                out_lines.append('')
            i = j
            continue

        # Lists (bulleted or numbered) — preserve line breaks between items
        if re.match(r"^\s*([-*+]\s+|\d+\.\s+)", line):
            out_lines.append(line)
            i += 1
            continue

        # Headings — keep and add a blank line after
        if stripped.startswith('#'):
            out_lines.append(line)
            if out_lines and out_lines[-1] != '':
                out_lines.append('')
            i += 1
            continue

        # Empty line — preserve a single empty line
        if stripped == '':
            if not out_lines or out_lines[-1] != '':
                out_lines.append('')
            i += 1
            continue

        # Regular paragraph line — append and ensure a blank line after to create paragraph spacing
        out_lines.append(line)
        if out_lines and out_lines[-1] != '':
            out_lines.append('')
        i += 1

    # Join and strip trailing whitespace/newlines
    formatted = '\n'.join(out_lines)
    # Collapse more than two consecutive newlines into two
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    return formatted.strip() + '\n'

def analyze_sentiment_and_get_llm_response(text):
    # Sentiment analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    # Determine textual label and badge color
    sentiment_text = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    badge_color = "#16a34a" if sentiment > 0 else "#dc2626" if sentiment < 0 else "#ffffff"
    # Choose a readable text color for the badge
    text_color = "#ffffff" if sentiment != 0 else "#000000"
    # Create a small rounded badge using inline HTML/CSS
    sentiment_label = (
        f'<span style="background:{badge_color}; color:{text_color}; '
        f'padding:4px 10px; border-radius:12px; border:1px solid #e5e7eb; '
        f'font-weight:600">{sentiment_text}</span>'
    )
    
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
    # Format the cleaned response for display (preserve tables/code blocks and paragraph spacing)
    formatted = format_llm_response(cleaned_response)
    return sentiment, sentiment_label, formatted
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
                # Display colored badge (HTML) and numeric score
                st.markdown(f"**Sentiment:** {sentiment_label}  \n**Score:** {sentiment:.2f}", unsafe_allow_html=True)
                
                if llm_response:
                    st.subheader("AI Response")
                    # Render LLM output as markdown/HTML so formatting is preserved
                    st.markdown(llm_response, unsafe_allow_html=True)

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
                    # Display colored badge (HTML) and numeric score
                    st.markdown(f"**Sentiment:** {sentiment_label}  \\  \n**Score:** {sentiment:.2f}", unsafe_allow_html=True)
                    
                    if llm_response:
                        st.subheader("AI Response")
                        # Render LLM output as markdown/HTML so formatting is preserved
                        st.markdown(llm_response, unsafe_allow_html=True)

                except sr.UnknownValueError:
                    st.error("Could not understand audio.")
                except sr.RequestError:
                    st.error("Could not request results from the speech recognition service.")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
