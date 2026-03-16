import streamlit as st
import validators
from parallel_summary import summarize_url_parallel


# ---------------- Page config ----------------

st.set_page_config(
    page_title="AI Web Page Summarizer",
    page_icon="🧠",
    layout="centered"
)


# ---------------- Cached function ----------------

@st.cache_data(show_spinner=False)
def run_summary(url, model, bullets, max_words):

    # auto worker tuning
    workers = 2 if model == "phi3:mini" else 1

    return summarize_url_parallel(
        url=url,
        model=model,
        bullets=bullets,
        max_words=max_words,
        workers=workers
    )


# ---------------- UI ----------------

st.title("🧠 AI Web Page Summarizer")

st.write(
    "Paste any webpage URL and generate a structured AI summary."
)


url = st.text_input(
    "Enter webpage URL",
    placeholder="https://example.com/article"
)


col1, col2 = st.columns(2)

with col1:
    bullets = st.slider(
        "Number of bullet points",
        3, 15, 6
    )

with col2:
    max_words = st.slider(
        "Summary length",
        150, 600, 300
    )


model = st.selectbox(
    "Select Model",
    ["phi3:mini", "gemma3:12b", "deepseek-r1"],
    help="phi3:mini is faster. Gemma is higher quality."
)


if st.button("Generate Summary", use_container_width=True):

    if not url:
        st.warning("Please enter a URL.")
        st.stop()

    if not validators.url(url):
        st.error("Invalid URL format.")
        st.stop()

    with st.spinner("Fetching page and generating summary..."):

        result = run_summary(url, model, bullets, max_words)
        st.write(result.get("abstract"))

    st.divider()

    st.subheader(result.get("title"))

    st.markdown(f"[Open Source Page]({url})")

    st.subheader("Summary")
    st.write(result.get("abstract"))

    st.subheader("Key Points")

    for b in result.get("bullets", []):
        st.markdown(f"- {b}")

    with st.expander("Raw JSON"):
        st.json(result)