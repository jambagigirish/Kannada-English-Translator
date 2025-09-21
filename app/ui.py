import streamlit as st
from models.translator import Translator


st.set_page_config(page_title="Kannada → English Translator")
st.title("Kannada → English Translator")


@st.cache_resource
def get_translator():
return Translator()


translator = get_translator()


kn_text = st.text_area("Enter Kannada text:", height=180)
if st.button("Translate"):
if kn_text.strip():
with st.spinner("Translating..."):
en_text = translator.translate(kn_text)
st.subheader("English:")
st.write(en_text)
else:
st.warning("Please enter some text.")
