import streamlit as st
from infer import Inference

st.title('Generate Abstracts From Titles of cs.CL Arxiv Papers')


st.subheader('Enter Title of Arxiv Paper (cs.CL):')
title = st.text_input(label='Title of Paper', placeholder='Type Here...', label_visibility='hidden')
st.subheader('Choose One Decoding Strategy for Text Generation:')
decoding_strategy = st.selectbox('Choose one decoding strategy',
                                 ('Greedy Search', 'Beam Search', 'Stochastic Sampling', 'Contrastive Search', 'Use Any'),
                                 label_visibility='hidden')
use_default = False
num_beams = 10
early_stopping = True
no_repeat_ngrams = 3
max_length = 1024
top_k = 50
top_p = 0.95
temperature = 0.7
alpha = 0.6

if decoding_strategy == 'Greedy Search':
    values = st.radio('**Choice of Hyperparameters:**', ('Use what the app maker set', 'Enter my own'))
    if values == 'Use what the app maker set':
        use_default = True
    else:
        no_repeat_ngrams = st.number_input('**No repeat n-gram size**', min_value=2)
        max_length = st.slider('**Max Length of Output Sequence Desired**', 0, 1024)

if decoding_strategy == 'Beam Search':
    values = st.radio('**Choice of Hyperparameters:**', ('Use what the app maker set', 'Enter my own'))
    if values == 'Use what the app maker set':
        use_default = True
    else:
        num_beams = st.number_input('**Enter no. of beams**', min_value=2, max_value=50000)
        early_stopping = st.checkbox('**Allow Early Stopping**')
        no_repeat_ngrams = st.number_input('**No repeat n-gram size**', min_value=2)
        max_length = st.slider('**Max Length of Output Sequence Desired**', 0, 1024)

elif decoding_strategy == 'Stochastic Sampling':
    values = st.radio('**Choice of Hyperparameters:**', ('Use what the app maker set', 'Enter my own'))
    if values == 'Use what the app maker set':
        use_default = True
    else:
        top_k = st.number_input('**top-k  (Set to 1 for only top-p sampling)**', 1, 50000)
        top_p = st.slider('**top-p  (Set to 1 for only top-k sampling)**', 0.0, 1.0, step=0.01)
        max_length = st.slider('**Max Length of Output Sequence Desired**', 0, 1024)
        temperature = st.number_input('**Temperature to control stochasticity**', 0.0, step=0.01)

elif decoding_strategy == 'Contrastive Search':
    values = st.radio('**Choice of Hyperparameters:**', ('Use what the app maker set', 'Enter my own'))
    if values == 'Use what the app maker set':
        use_default = True
    else:
        top_k = st.number_input('**top-k**', 2, 50000)
        max_length = st.slider('**Max Length of Output Sequence Desired**', 0, 1024)
        alpha = st.slider('**Penalty Alpha (Value of 1 reverts to greedy search)**', 0.0, 1.0, 0.01)

submit = st.button('Generate Abstract:')
if submit:
    gen = Inference()
    abstract = ''
    with st.spinner("Generating Results..."):
        if decoding_strategy == 'Use Any':
            abstract = gen.generate_abstract(title=title)
        elif use_default:
            abstract = gen.generate_abstract(title=title, decoding_strategy=decoding_strategy)
        else:
            abstract = gen.generate_abstract(title=title,
                                             decoding_strategy=decoding_strategy,
                                             num_beams=num_beams,
                                             early_stopping=early_stopping,
                                             no_repeat_ngram_size=no_repeat_ngrams,
                                             max_length=max_length,
                                             top_k=top_k,
                                             top_p=top_p,
                                             temperature=temperature,
                                             penalty_alpha=alpha
                                             )
    abstract = abstract.split('<|SEP|>')[1]
    st.write(f'**Generated Abstract:**\n {abstract}')
