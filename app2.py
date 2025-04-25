import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import io
from pydub import AudioSegment
from elevenlabs import ElevenLabs


# --- Configuration ---
TOXICITY_MODEL_NAME = "unitary/toxic-bert"


# --- Caching Toxicity Model ---
@st.cache_resource
def load_toxicity_model():
    """Loads the Text Classification model and tokenizer."""
    print(f"Loading toxicity model: {TOXICITY_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_NAME)
    print("Toxicity model loaded successfully!")
    return model, tokenizer


# --- Prediction Functions ---
def predict_toxicity(model, tokenizer, text):
    """Predicts toxicity scores for the given text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
        return_attention_mask=True,
    )

    if (
        model.parameters()
        and next(model.parameters(), None) is not None
        and next(model.parameters()).is_cuda
    ):
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)

    predictions = predictions.cpu().numpy()[0]
    labels = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
    ]
    results = {label: float(pred) for label, pred in zip(labels, predictions)}
    return results


def transcribe_audio_elevenlabs(api_key, audio_bytes):
    """Transcribes audio bytes to text using the Eleven Labs STT API."""
    # 1. Read audio bytes using pydub
    audio_stream = io.BytesIO(audio_bytes)
    audio_segment = AudioSegment.from_file(audio_stream)

    # 2. Convert to WAV format in memory
    wav_stream = io.BytesIO()
    audio_segment.export(wav_stream, format="wav")
    wav_stream.seek(0)

    # 3. Initialize Eleven Labs client and send audio
    client = ElevenLabs(api_key=api_key)

    # 4. Sending data to eleven labs for transcription
    response = client.speech_to_text.convert(
        file=(f"recording.wav", wav_stream),
        model_id="scribe_v1",
        language_code="eng",
    )

    transcription = response.text

    return transcription.strip()


# --- Main Streamlit App ---
def main():
    # Page configuration
    st.set_page_config(
        page_title="Toxic Content Detection", page_icon="🛡️", layout="centered"
    )

    # Header
    st.title("🛡️ Toxic Content Detection")
    st.markdown(
        """
    Analyze text for toxicity, or record audio to transcribe and analyze using Eleven Labs.
    Supports detection of:
    - Toxicity - Severe Toxicity - Obscene Language - Threats - Insults - Identity Attacks
    """
    )
    toxicity_model, toxicity_tokenizer = load_toxicity_model()

    # Get Eleven Labs API Key from Streamlit secrets
    elevenlabs_api_key = st.secrets.get("ELEVENLABS_API_KEY")

    # --- Initialize Session State for Text ---
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "last_input_type" not in st.session_state:
        st.session_state.last_input_type = "text"

    # --- Input Methods ---
    tab1, tab2 = st.tabs(["📝 Text Input", "🎤 Voice Input (Eleven Labs)"])

    # --- Text Input Tab ---
    with tab1:
        text_area_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here...",
            key="text_area",
            value=(
                st.session_state.text_input
                if st.session_state.get("last_input_type") != "voice"
                else ""
            ),
        )
        if text_area_input != st.session_state.text_input and text_area_input.strip():
            st.session_state.text_input = text_area_input
            st.session_state.last_input_type = "text"
            if "analysis_results" in st.session_state:
                st.session_state.analysis_results = None

    # --- Voice Input Tab ---
    with tab2:
        st.write("Click the microphone icon to start/stop recording:")
        audio_info = None
        from streamlit_mic_recorder import mic_recorder

        audio_info = mic_recorder(
            start_prompt="⏺️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            key="voice_recorder_elevenlabs",
        )

        if audio_info and audio_info.get("bytes"):
            st.audio(audio_info["bytes"], format="audio/wav")
            audio_bytes = audio_info["bytes"]
            with st.spinner("Sending audio to Eleven Labs for transcription..."):
                transcribed_text = transcribe_audio_elevenlabs(
                    elevenlabs_api_key, audio_bytes
                )

            if transcribed_text and not transcribed_text.startswith(
                "[Transcription Error"
            ):
                st.session_state.text_input = transcribed_text
                st.session_state.last_input_type = "voice"
                st.success(
                    "Transcription complete! Text populated below and in the 'Text Input' tab."
                )
                st.session_state.analysis_results = None

            elif transcribed_text.startswith("[Transcription Error"):
                st.warning(f"Transcription failed: {transcribed_text}")
            else:
                st.warning("Transcription resulted in empty text.")

        edited_text = st.text_area(
            "Transcribed/Current Text (edit if needed):",
            value=st.session_state.text_input,
            key="transcribed_or_current_text_area",
            height=100,
        )

        if edited_text != st.session_state.text_input:
            st.session_state.text_input = edited_text
            st.session_state.last_input_type = "edited"
            if "analysis_results" in st.session_state:
                st.session_state.analysis_results = None

    # --- Analysis Section ---
    st.divider()

    if st.button("Analyze Text", key="analyze_button_elevenlabs"):
        text_to_analyze = st.session_state.get("text_input", "").strip()

        if not text_to_analyze:
            st.warning("⚠️ Please enter some text or record audio first.")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    results = predict_toxicity(
                        toxicity_model, toxicity_tokenizer, text_to_analyze
                    )
                    st.session_state.analysis_results = results

                    print("\n=== Model Results ===")
                    print(f"Input Text: {text_to_analyze}")
                    print("Predictions:")
                    for label, score in results.items():
                        print(f"{label}: {score:.4f}")
                    print("==================\n")

                except Exception as analysis_error:
                    st.error(f"Error during toxicity analysis: {analysis_error}")
                    print(f"Analysis Error: {analysis_error}")
                    st.session_state.analysis_results = None

    # --- Display Results ---
    if st.session_state.get("analysis_results"):
        results = st.session_state.analysis_results
        st.subheader("Analysis Results")
        cols = st.columns(2)
        for idx, (label, score) in enumerate(results.items()):
            with cols[idx % 2]:
                if score < 0.3:
                    color, severity = "green", "Low"
                elif score < 0.7:
                    color, severity = "orange", "Moderate"
                else:
                    color, severity = "red", "High"
                st.markdown(
                    f"""
                    <div style='margin-bottom: 10px; padding: 10px; border-radius: 5px; border: 1px solid {color}'>
                    <b>{label.replace('_', ' ').title()}:</b><br>
                    Score: {score:.2%}<br>
                    Severity: <span style='color:{color};'>{severity}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("Overall Assessment")
        max_toxicity = max(results.values())
        if max_toxicity < 0.3:
            st.success("✅ Generally safe.")
        elif max_toxicity < 0.7:
            st.warning("⚠️ Potentially problematic content.")
        else:
            st.error("🚫 Likely toxic content.")

        st.info(
            """
            💡 **Interpretation Guide:** Low: <30%, Moderate: 30-70%, High: >70%.
            AI models are not perfect; consider context.
            """
        )

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center'>
        <p>Built with Streamlit & Hugging Face 🤗</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
