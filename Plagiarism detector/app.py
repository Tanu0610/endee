import streamlit as st
from embeddings import get_embedding
from database import init_index, store_document, get_all_documents
from similarity import check_plagiarism

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Plagiarism Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Plagiarism Detector")
st.markdown("*Powered by Endee Vector Database + Sentence Transformers*")
st.divider()

# ─── Initialize Endee Index ────────────────────────────────────────────────────
@st.cache_resource
def setup():
    return init_index()

index = setup()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("📚 Document Library")
docs = get_all_documents()
if docs:
    st.sidebar.success(f"{len(docs)} document(s) stored in Endee")
    for doc in docs:
        st.sidebar.markdown(f"- 📄 `{doc}`")
else:
    st.sidebar.info("No documents stored yet. Add some below!")

# ─── Layout: Two Columns ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

# ─── Column 1: Add Original Documents ─────────────────────────────────────────
with col1:
    st.subheader("➕ Add Original Document")
    st.markdown("Store original documents in Endee vector database.")

    doc_title = st.text_input("Document Title", placeholder="e.g. Research Paper 1")
    doc_text = st.text_area(
        "Document Content",
        height=200,
        placeholder="Paste your original document text here..."
    )

    if st.button("📥 Store in Endee", use_container_width=True):
        if doc_title.strip() and doc_text.strip():
            with st.spinner("Generating embedding and storing in Endee..."):
                embedding = get_embedding(doc_text)
                store_document(index, doc_title, doc_text, embedding)
            st.success(f"✅ '{doc_title}' stored successfully in Endee!")
            st.rerun()
        else:
            st.error("Please provide both a title and content.")

# ─── Column 2: Check for Plagiarism ───────────────────────────────────────────
with col2:
    st.subheader("🔎 Check for Plagiarism")
    st.markdown("Paste text to check if it matches any stored documents.")

    check_text = st.text_area(
        "Text to Check",
        height=200,
        placeholder="Paste the text you want to check for plagiarism..."
    )

    threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=30,
        max_value=95,
        value=70,
        help="Texts above this similarity score will be flagged as plagiarised."
    )

    if st.button("🚀 Check Plagiarism", use_container_width=True, type="primary"):
        if check_text.strip():
            if not get_all_documents():
                st.warning("⚠️ No documents stored yet. Please add original documents first.")
            else:
                with st.spinner("Analyzing text with Endee vector search..."):
                    results = check_plagiarism(index, check_text, threshold / 100)

                st.divider()
                st.subheader("📊 Results")

                if results:
                    top = results[0]
                    score_pct = round(top["similarity"] * 100, 2)

                    # Verdict
                    if score_pct >= threshold:
                        st.error(f"❌ PLAGIARISM DETECTED! Similarity: **{score_pct}%**")
                    elif score_pct >= 50:
                        st.warning(f"⚠️ PARTIALLY SIMILAR. Similarity: **{score_pct}%**")
                    else:
                        st.success(f"✅ ORIGINAL CONTENT. Similarity: **{score_pct}%**")

                    # Score bar
                    st.progress(score_pct / 100)

                    # All matches table
                    st.markdown("#### 🔗 All Matches Found")
                    for i, r in enumerate(results):
                        pct = round(r["similarity"] * 100, 2)
                        label = "🔴" if pct >= threshold else "🟡" if pct >= 50 else "🟢"
                        with st.expander(f"{label} Match #{i+1} — `{r['title']}` — {pct}% similar"):
                            st.markdown(f"**Document:** {r['title']}")
                            st.markdown(f"**Similarity Score:** `{pct}%`")
                            st.markdown(f"**Stored Content Preview:**")
                            st.info(r["content"][:500] + "..." if len(r["content"]) > 500 else r["content"])
                else:
                    st.success("✅ No similar documents found. Content appears to be original!")
        else:
            st.error("Please enter some text to check.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center>Built with ❤️ using Endee Vector DB + Sentence Transformers + Streamlit</center>",
    unsafe_allow_html=True
)
