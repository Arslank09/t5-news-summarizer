import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: white; }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📰 News Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any news article — AI will summarize it! • Fine-tuned T5 • By Arslan Kareem</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

with st.spinner("Loading AI model..."):
    tokenizer, model, device = load_model()

st.success("Model ready! ✅")

article = st.text_area(
    "Paste your news article here:",
    height=300,
    placeholder="Paste any news article here..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    summarize_btn = st.button("⚡ Summarize!", use_container_width=True)

if summarize_btn:
    if article.strip():
        with st.spinner("Summarizing..."):
            input_text = "summarize: " + article
            inputs = tokenizer(input_text, return_tensors="pt",
                             max_length=512, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown("### 📝 Summary:")
        st.success(summary)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Length", f"{len(article.split())} words")
        with col2:
            st.metric("Summary Length", f"{len(summary.split())} words")
    else:
        st.error("Please paste an article first!")

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("This app uses a **fine-tuned T5** model trained on CNN/DailyMail dataset.")
    st.markdown("---")
    st.markdown("**🛠️ Tech Stack:**")
    st.markdown("- 🤗 Hugging Face Transformers")
    st.markdown("- 🧠 T5-small (Fine-tuned)")
    st.markdown("- 🎨 Streamlit")
    st.markdown("- 🔥 PyTorch")
    st.markdown("---")
    st.markdown("**👨‍💻 Built by Arslan Kareem**")
    st.markdown("BSCS — Federal Urdu University")