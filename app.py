import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import io
import numpy as np
from pydub import AudioSegment

TOXICITY_MODEL_NAME = "unitary/toxic-bert"
ASR_MODEL_NAME = "openai/whisper-large-v3-turbo"
TARGET_SAMPLE_RATE = 16000


@st.cache_resource
def load_toxicity_model():
    """Loads the Text Classification model and tokenizer."""
    print(f"Loading toxicity model: {TOXICITY_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_NAME)
    print("Toxicity model loaded successfully!")
    return model, tokenizer


@st.cache_resource
def load_asr_model():
    """Loads the Automatic Speech Recognition model and processor."""
    print(f"Loading ASR model: {ASR_MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_NAME,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    print(f"ASR model loaded successfully on {device}!")
    return processor, model, device


def predict_toxicity(model, tokenizer, text):
    """Predicts toxicity scores for the given text."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    if next(model.parameters()).is_cuda:
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


def transcribe_audio(asr_processor, asr_model, device, audio_bytes):
    """Transcribes audio bytes to text using the ASR model."""
    if not audio_bytes:
        return ""
    try:
        audio_stream = io.BytesIO(audio_bytes)
        try:
            audio_segment = AudioSegment.from_file(audio_stream)
            print(
                f"pydub successfully loaded audio. Frame rate: {audio_segment.frame_rate}, Channels: {audio_segment.channels}"
            )
        except Exception as pydub_error:
            st.error(
                f"pydub failed to load audio: {pydub_error}. Ensure ffmpeg is installed and in PATH."
            )
            print(f"pydub failed: {pydub_error}")
            return "[Transcription Error: pydub load failed]"

        wav_stream = io.BytesIO()
        audio_segment.export(wav_stream, format="wav")
        wav_stream.seek(0)

        speech_array, sampling_rate = librosa.load(wav_stream, sr=TARGET_SAMPLE_RATE)

        speech_array = librosa.to_mono(speech_array)
        if speech_array.dtype != np.float32:
            speech_array = speech_array.astype(np.float32)
        print(
            f"Librosa successfully loaded WAV. Original SR (from WAV): {sampling_rate}, Target SR: {TARGET_SAMPLE_RATE}, Array shape: {speech_array.shape}"
        )

        input_features = asr_processor(
            speech_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_features
        input_features = input_features.to(device)
        if asr_model.dtype == torch.float16:
            input_features = input_features.half()

        with torch.no_grad():
            if isinstance(asr_model, AutoModelForSpeechSeq2Seq):
                predicted_ids = asr_model.generate(input_features, max_new_tokens=128)
            else:
                st.error("Unsupported ASR model type for generation.")
                return "[Transcription Error: Model type]"

        transcription = asr_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        print(f"Transcription: {transcription}")
        return transcription.strip()

    except Exception as e:
        st.error(f"Error during transcription: {e}")
        import traceback

        print("--- Transcription Error Traceback ---")
        traceback.print_exc()
        print("------------------------------------")
        return "[Transcription Error]"


def main():
    st.set_page_config(
        page_title="Toxic Content Detection", page_icon="🛡️", layout="centered"
    )

    st.title("🛡️ Toxic Content Detection")
    st.markdown(
        """
    Analyze text for toxicity, or record audio to transcribe and analyze.
    Supports detection of:
    - Toxicity - Severe Toxicity - Obscene Language - Threats - Insults - Identity Attacks
    """
    )
    st.info(
        "🚨 **Note:** Audio processing requires `ffmpeg` installed on your system and available in your PATH."
    )

    try:
        toxicity_model, toxicity_tokenizer = load_toxicity_model()
        asr_processor, asr_model, asr_device = load_asr_model()
    except Exception as e:
        error_msg = f"😕 Oops! Could not load models: {str(e)}"
        print(f"\nERROR: {error_msg}\n")
        st.error(error_msg)
        st.stop()

    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "last_input_type" not in st.session_state:
        st.session_state.last_input_type = "text"  # or 'voice'

    tab1, tab2 = st.tabs(["📝 Text Input", "🎤 Voice Input"])

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
        if text_area_input != st.session_state.text_input and text_area_input:
            st.session_state.text_input = text_area_input
            st.session_state.last_input_type = "text"
            if "analysis_results" in st.session_state:
                st.session_state.analysis_results = None

    with tab2:
        st.write("Click the microphone icon to start/stop recording:")
        audio_info = None  # Initialize
        try:
            from streamlit_mic_recorder import mic_recorder

            audio_info = mic_recorder(
                start_prompt="⏺️ Start Recording",
                stop_prompt="⏹️ Stop Recording",
                key="voice_recorder",
            )
        except ImportError:
            st.error(
                "`streamlit-mic-recorder` not installed. Please run `pip install streamlit-mic-recorder`"
            )
        except Exception as e:
            st.error(f"Error initializing microphone recorder: {e}")

        if audio_info and audio_info.get("bytes"):
            st.audio(audio_info["bytes"], format="audio/wav")
            audio_bytes = audio_info["bytes"]
            with st.spinner("Transcribing audio... Please wait."):
                transcribed_text = transcribe_audio(
                    asr_processor, asr_model, asr_device, audio_bytes
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

    st.divider()

    if st.button("Analyze Text", key="analyze_button"):
        text_to_analyze = st.session_state.get("text_input", "").strip()

        if not text_to_analyze:
            st.warning("⚠️ Please enter some text or record audio first.")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    results = predict_toxicity(
                        toxicity_model, toxicity_tokenizer, text_to_analyze
                    )
                    st.session_state.analysis_results = results  # Store results
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
        <p><small>ASR: {ASR_MODEL_NAME} | Toxicity: {TOXICITY_MODEL_NAME}</small></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
