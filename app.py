import streamlit as st
from model_utils import load_resources, generate_summary

#adjust model paths 
MODEL_PATH = 'all_outputs/seq2seq_attention.keras'
TOKENIZER_PATH = 'all_outputs/tokenizer.pkl'
CONFIG_PATH = 'all_outputs/config.pkl'



@st.cache_resource
def load_model_once():
    return load_resources(MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH)


st.set_page_config(page_title='Arabic Summarization', layout='centered')
st.title('Arabic Text Summarization')
st.caption('Seq2Seq with Bahdanau Attention')

with st.spinner('Loading model...'):
    train_model, encoder_model, decoder_model, tokenizer, config = load_model_once()

article = st.text_area('Article', height=250, placeholder='Paste Arabic article here...')

if st.button('Generate Summary'):
    if not article.strip():
        st.warning('Please enter some text.')
    else:
        with st.spinner('Generating...'):
            summary = generate_summary(
                article, encoder_model, decoder_model, tokenizer, config
            )
        st.subheader('Summary')
        st.write(summary if summary else '(empty output)')