import streamlit as st
from transformers import pipeline

def main():
    # Set up the page configuration
    st.set_page_config(
        page_title="Hate Speech Detection",
        page_icon="🔍",
        layout="wide"
    )

    # Create header
    st.title("Hate Speech Detection App")
    st.markdown("""
    This app analyzes text for different types of toxic content including:
    - Toxic
    - Severe Toxic
    - Obscene
    - Threat
    - Insult
    - Identity Hate
    """)

    # Initialize the pipeline
    @st.cache_resource
    def load_model():
        return pipeline("text-classification", 
                       model="devesh1011/fine-tuned-bert-hate-speech-model")

    # Load the model
    pipe = load_model()

    # Create text input
    text_input = st.text_area(
        "Enter your text here:",
        height=200,
        placeholder="Paste your text or article here..."
    )

    # Add analyze button
    if st.button("Analyze Text"):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Get prediction
                result = pipe(text_input)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create a color-coded box based on the prediction
                label = result[0]['label']
                score = result[0]['score']
                
                # Format the score as percentage
                score_percentage = f"{score:.2%}"
                
                # Display result with color coding
                if label == "LABEL_0":
                    st.success(f"✅ This text appears to be NON-TOXIC (Confidence: {score_percentage})")
                else:
                    st.error(f"⚠️ This text appears to be TOXIC (Confidence: {score_percentage})")
                
                # Display detailed explanation
                st.info("""
                Note: This analysis is based on machine learning and may not be 100% accurate. 
                The model evaluates the text for various forms of toxic content including hate speech, 
                threats, obscenity, insults, and identity-based attacks.
                """)

if __name__ == "__main__":
    main() 