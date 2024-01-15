import os
import json
import re

import streamlit as st
import streamlit.components.v1 as components
import stardog
import torch
import transformers

from torch import bfloat16
from typing import Iterator
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from Augur.model import get_input_token_length, run
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from rdflib import Graph

import Augur.model as augur_model

load_dotenv()

SD_USER = os.getenv('SD_USER')
SD_PASSWORD= os.getenv('SD_PASSWORD')

MAX_MAX_NEW_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 512
MAX_INPUT_TOKEN_LENGTH = 4000

st.set_page_config(layout="wide")
st.title('Augur: Machine Query Translation')


st.markdown("""
<style>
    div[data-baseweb="textarea"] > div {
        height: 20vh !important;
    }
</style>
""", unsafe_allow_html=True)


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    history = history_with_input[:-1]
    generator = run(message, history, system_prompt, model, tokenizer, max_new_tokens, temperature, top_p, top_k)
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, '')]
    for response in generator:
        yield history + [(message, response)]


def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    input_token_length = get_input_token_length(message, chat_history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise Exception(f'The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')
    

def send_consult(query):
    conn_details = {
    'endpoint': 'https://sd-0d6bd678.stardog.cloud:5820',
    'username': SD_USER,
    'password': SD_PASSWORD
    }

    # Initialize connection
    with stardog.Connection('PD_KG', **conn_details) as conn:        
        # Execute the query
        if 'CONSTRUCT' in query:
            results = conn.graph(query)
        else:
            results = conn.select(query)

    return results

@st.cache_resource()
def load_vector_db():
    graph = Graph()
    graph.parse('./datafiles/ontology.ttl', format='ttl')

    emb_model = "sentence-transformers/all-mpnet-base-v2"
    #model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)#, model_kwargs=model_kwargs)
    
    loader = TextLoader('./datafiles/ontology.jsonld')
    data = loader.load()
    db = Chroma.from_texts([object for _,_,object in graph],
                           embedding=embeddings,
                           metadatas=[{'subj': subj, 'pred':pred, 'obj': obj} for subj,pred, obj in graph],
                           persist_directory='./db')
    
    return db

# Load vector database
v_db = load_vector_db()

def chat_model():
    model, tokenizer = augur_model.load_quant_model('codellama/CodeLlama-13b-Instruct-hf')
    return model, tokenizer

with st.sidebar:
    st.subheader('Model parameters')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=2.0, value=1., step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=1, max_value=100, value=10, step=5)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    model, tokenizer = chat_model()
    st.subheader('In-context Learning')
    cove = st.checkbox('Chain of Verification')
    cot = st.checkbox('Chain of Thought')
    rag = st.checkbox('Retrieval Augmented Generation')
    keys = st.checkbox('Foreign Keys')
    few_shot = st.checkbox('Few-shot Learning')
    agents = st.checkbox('Multi-Agents and ensemble')

# User-provided prompt

if prompt := st.chat_input():
    with st.container():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

col1, col2 = st.columns([1, 1])

if 'output' not in st.session_state:
    st.session_state.output = ''

if 'first_message' not in st.session_state:
    st.session_state.first_message = True

with col1:
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Make a database request in natural language:"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Make a database request in natural language:"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Generate a new response if last message is not from assistant

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = v_db.similarity_search(prompt, include_metadata=True, k=5)
                
                if not rag: retrieved = ''
                if not few_shot: DEFAULT_SYSTEM_PROMPT = ''
                if not keys: test = ''

                message = test + CONTEXT + str(retrieved) + '.' + prompt + PROMPT_END
                st.session_state.first_message = False

                response = generate(message, 
                                    [''], 
                                    DEFAULT_SYSTEM_PROMPT, 
                                    model,
                                    tokenizer,
                                    MAX_MAX_NEW_TOKENS, 
                                    temperature, 
                                    top_p, 
                                    top_k,
    )
                placeholder = st.empty()
                full_response = ''

                for item in response:
                    placeholder.markdown(item[-1][1])
                #placeholder.markdown(full_response[-1])
                    full_response = item[-1][1]
                code_pattern = r'```(.*?)```'
                code = re.findall(code_pattern, full_response, re.DOTALL)

                st.write('**Debug: Código capturado**')
                code_box = st.empty()
                st.session_state.output = next(iter(code))
                output = f'```{st.session_state.output}```'
                code_box.markdown(output)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

with col2: 
    st.write("CAPTURED CONSULT")
    code_box = st.empty()
    code_box.markdown(f'```{st.session_state.output + "LIMIT 10"}')
    if st.button('SEND CONSULT'):
        consult_output = st.empty()
        json_string = send_consult((st.session_state.output if 'output' in st.session_state else '') + 'LIMIT 10')
        try: 
            json_data = json.dumps(json_string, indent=4) 
            json_string = json_data
        except:
            print('Graph output')  
        
        consult_output.markdown(f"```json\n{json_string}\n```")
    else:
        st.write('Triplestore consult results')


