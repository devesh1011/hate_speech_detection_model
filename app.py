import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_model():
    # Load model and tokenizer from Unitary's toxic-bert
    model_name = "unitary/toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def predict(model, tokenizer, text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
    
    # Convert to numpy for easier handling
    predictions = predictions.numpy()[0]
    
    # Define the labels for toxic-bert
    labels = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack"
    ]
    
    # Create results dictionary
    results = {label: float(pred) for label, pred in zip(labels, predictions)}
    return results

def main():
    # Page configuration
    st.set_page_config(
        page_title="Toxic Comment Detection",
        page_icon="🛡️",
        layout="centered"
    )

    # Header
    st.title("🛡️ Toxic Comment Detection")
    st.markdown("""
    This application analyzes text for different types of toxic content including:
    - Toxicity
    - Severe Toxicity
    - Obscene Language
    - Threats
    - Insults
    - Identity Attacks
    """)

    # Load model
    try:
        model, tokenizer = load_model()
        print("Model loaded successfully!")

        # Text input
        user_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )

        # Analysis button
        if st.button("Analyze Text"):
            if not user_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing text..."):
                    # Get predictions
                    results = predict(model, tokenizer, user_input)
                    
                    # Print results to terminal
                    print("\n=== Model Results ===")
                    print(f"Input Text: {user_input}")
                    print("Predictions:")
                    for label, score in results.items():
                        print(f"{label}: {score:.4f}")
                    print("==================\n")
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Create columns for results
                    cols = st.columns(2)
                    
                    # Display each category with its probability
                    for idx, (label, score) in enumerate(results.items()):
                        with cols[idx % 2]:
                            # Determine color based on score
                            if score < 0.3:
                                color = "green"
                                severity = "Low"
                            elif score < 0.7:
                                color = "orange"
                                severity = "Moderate"
                            else:
                                color = "red"
                                severity = "High"
                            
                            # Display score with formatting
                            st.markdown(
                                f"""
                                <div style='padding: 10px; border-radius: 5px; border: 1px solid {color}'>
                                <b>{label.replace('_', ' ').title()}:</b><br>
                                Score: {score:.2%}<br>
                                Severity: {severity}
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    
                    # Overall assessment
                    st.subheader("Overall Assessment")
                    max_toxicity = max(results.values())
                    if max_toxicity < 0.3:
                        st.success("✅ This text appears to be generally safe and non-toxic.")
                    elif max_toxicity < 0.7:
                        st.warning("⚠️ This text contains potentially problematic content.")
                    else:
                        st.error("🚫 This text contains highly toxic content.")
                    
                    # Add interpretation note
                    st.info("""
                    💡 **Interpretation Guide:**
                    - Green (Low): Less than 30% probability of toxic content
                    - Orange (Moderate): 30-70% probability of toxic content
                    - Red (High): More than 70% probability of toxic content
                    
                    Note: This is an AI model and may not be 100% accurate. Consider context and nuance when interpreting results.
                    """)

    except Exception as e:
        error_msg = f"😕 Oops! Something went wrong: {str(e)}"
        print(f"\nERROR: {error_msg}\n")
        st.error(error_msg)

    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>Built with Streamlit and Hugging Face Transformers 🤗</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()