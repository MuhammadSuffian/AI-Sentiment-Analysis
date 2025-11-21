# AI Sentiment Analysis

A simple Streamlit app that transcribes audio, performs sentiment analysis using TextBlob, and returns an LLM-driven empathetic response via the Groq client.

**Files**
- `sentiment_analysis.py`: main Streamlit app.
- `requirements.txt`: Python dependencies.
- `.python-version` (optional): suggested Python runtime.

**Features**
- Upload audio or record live audio in the browser.
- Speech-to-text via `SpeechRecognition` (uses Google Web Speech API by default).
- Sentiment analysis with `TextBlob`.
- LLM response using the `groq` client.

**Requirements**
- Python 3.11 recommended (see notes below for Cloud compatibility).
- Install dependencies with `pip` using the repository `requirements.txt`.

Installation (local):

```powershell
# Create and activate a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

Run the app locally:

```powershell
streamlit run sentiment_analysis.py
```

Deployment notes
- If deploying to a cloud service (Streamlit Cloud, other hosts), prefer Python 3.11 to avoid compatibility issues with some audio packages. You can add a `.python-version` or `runtime.txt` with `3.11` to request that runtime on platforms that respect those files.
- Streamlit Cloud and other hosts may use newer Python versions (3.13+) by default; if you see import errors for `aifc` or similar, switch to Python 3.11.

Troubleshooting
- Module import errors referencing `aifc` or errors while installing `pyaudio` / `playsound` typically indicate runtime mismatch (e.g., Python 3.13). Recommended actions:
  - Use Python 3.11 (create a `.python-version` file with `3.11` or add `runtime.txt` with `python-3.11` for platforms that support it).
  - Avoid `pyaudio` and `playsound` on Cloud environments if they fail to build; instead:
    - Use Streamlit's `st.audio()` or browser-based recording (`st.audio_input`) for playback/recording.
    - Use `pydub` + `simpleaudio` if you need server-side audio processing and the host supports building wheels.
- If `SpeechRecognition` raises errors under Python 3.13, pin to a compatible runtime or remove the package and rely on browser-side recording + a cloud speech-to-text API.

Notes
- The app expects a Groq API key stored in Streamlit secrets under the key `api_tokken` (note the variable name used in the code). Add this via your deployment platform's secrets configuration.

Author
- Repository owner: MuhammadSuffian

If you want, I can also:
- Add a `runtime.txt` and `.python-version` to explicitly request Python 3.11 in the repo.
- Pin exact package versions in `requirements.txt` after verifying the environment's installed versions. 