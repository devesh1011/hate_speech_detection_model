import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

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
    This app analyzes text for toxic content and provides classification for:
    - Toxic
    - Severe Toxic
    - Obscene
    - Threat
    - Insult
    - Identity Hate
    """)

    # Initialize the model and tokenizer
    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained("devesh1011/Upload_tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased",
            num_labels=6
        )
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )

    # Load the model
    try:
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
    
    except Exception as e:
        st.error(f"""
        Error loading the model. Please try again later.
        If the problem persists, contact support.
        Technical details: {str(e)}
        """)

if __name__ == "__main__":
    main() 