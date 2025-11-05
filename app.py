import streamlit as st
import spacy
from spacy import displacy
import urllib
import pandas as pd

# --- Load spaCy model once ---
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# --- App Layout ---
st.title("⚖️ Basic NER - Judgment Entity Extractor")
st.write("Upload or paste a court judgment text to extract named entities like PERSON, ORG, DATE, etc.")

option = st.radio("Choose input type", ["Paste Text", "Upload File", "Use Sample Judgment"])

if option == "Paste Text":
    judgment_text = st.text_area("Paste Judgment Text Here", height=300)

elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload .txt file", type="txt")
    if uploaded_file:
        judgment_text = uploaded_file.read().decode("utf-8")
    else:
        judgment_text = ""

else:
    st.write("Using sample judgment from GitHub...")
    url = "https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment1.txt"
    judgment_text = urllib.request.urlopen(url).read().decode()

if st.button("🔍 Extract Entities"):
    if judgment_text.strip() == "":
        st.warning("Please provide a judgment text.")
    else:
        with st.spinner("Extracting entities... please wait"):
            doc = nlp(judgment_text)

        st.success("Extraction complete ✅")

        # --- Display Entities ---
        st.subheader("Extracted Entities")
        ent_data = [{"Entity": ent.text, "Label": ent.label_} for ent in doc.ents]
        st.dataframe(ent_data)

        # --- Visualization using displacy ---
        colors = {
            'PERSON': "#f4a261",
            'ORG': "#2a9d8f",
            'GPE': "#e9c46a",
            'DATE': "#e76f51",
            'LAW': "#264653"
        }
        options = {"ents": list(set([e.label_ for e in doc.ents])), "colors": colors}
        html = displacy.render(doc, style="ent", options=options)
        st.components.v1.html(html, height=600, scrolling=True)

        # --- Download as CSV ---
        csv = pd.DataFrame(ent_data).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Entities as CSV", csv, "entities.csv", "text/csv")

