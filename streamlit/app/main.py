# /app/main.py
import torch, os
from importlib.metadata import version
import streamlit as st
import tiktoken
from pathlib import Path
from scripts import MultiHeadAttention, LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, build_old_policy

# library = ["numpy", "torch", "tensorflow", "streamlit", "pandas", "tiktoken"]
# for lib in library:
#     st.write(f"{lib} version: {version(lib)}")

# Set basic page configuration (optional, but good for wider layouts)
st.set_page_config(
    page_title="Spam or Ham",
    page_icon="🤖",
    layout="centered", # or "wide"
    initial_sidebar_state="collapsed"
)

# BUILD THE CLASSIFIER POLICY MODEL
@st.cache_resource
def load_model_and_tokenizer():
    # --- CONFIGURATION ---
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    policy = build_old_policy(base_config=BASE_CONFIG, chosen_model="gpt2-small (124M)", num_classes=2)
    
    model_parameters_path= Path("./app/models/Spam-Classifier-GPT2-Model.pt")   # Factor in that the docker image will start in a different working directory (see Dockerfile)

    if not model_parameters_path.exists():
        st.error(f"Model Parameter file not found at: {model_parameters_path}. Please ensure it's in the correct location.")
        st.stop() # Stop the script


    policy.load_state_dict(torch.load(f=model_parameters_path, weights_only=True, map_location='cpu'))
    policy.to('cpu')
    tokenizer = tiktoken.get_encoding("gpt2")

    return policy.eval(), tokenizer

st.title("Spam Classifier Agent!")

# https://docs.streamlit.io/develop/api-reference/widgets/st.text_area
text_block = st.text_area(label="Enter your text to classify if it is SPAM or NOT SPAM", placeholder ="ConGratulations!!!1 You won $1.000. Click the link beelow to claime you're Prize.!")


# --- Add a button to trigger analysis ---
if st.button("Analyze Text"):
    if text_block:  # Run if there is an input ; maybe introduce a 'submit' button

        policy, tokenizer = load_model_and_tokenizer()

        # Tokenize the input string and restrict it to the model's context length
        tokenized_input = tokenizer.encode(text_block)[-policy.pos_emb.num_embeddings:]

        batched_input = torch.tensor(data=tokenized_input).unsqueeze(0)  # turn the tokenized input into a tensor and add a batch dimension
        with torch.no_grad():
            logits = policy(batched_input)[:,-1,:]   # Run the logits through the model and extract the probabilities of the last timestep

        prediction_index = torch.argmax(input=logits, dim=-1).item()  # Get the prediction of the model

        prediction_label = "SPAM" if prediction_index == 1 else "NOT SPAM"  # Map the prediction index to a label
        
        # --- Streamlit Output ---
        st.subheader("Classification Result:")
        st.write("---") # Separator

        st.markdown(f"**Classification:**")
        if prediction_label == "SPAM":
            st.error(f"Prediction: {prediction_label} 🚨") # Red box for spam
        else:
            st.success(f"Prediction: {prediction_label} ✅") # Green box for not spam

        # Optional: Show probabilities
        softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
        st.info(f"Probabilities: SPAM={softmax_probs[0, 1]:.4f}, NOT SPAM={softmax_probs[0, 0]:.4f}")

        st.write("---") # Another separator
    else:
        st.warning("Please enter some text in the text area before clicking 'Analyze Text'.")
