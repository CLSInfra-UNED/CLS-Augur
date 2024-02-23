import os
import json
import re

import pandas as pd 

import streamlit as st

from dotenv import load_dotenv
from threading import Thread

import Augur.context_templates as ctx
from Augur import (model, rag)

load_dotenv()

SD_USER = os.getenv('SD_USER')
SD_PASSWORD= os.getenv('SD_PASSWORD')

MODEL_ID = 'codellama/CodeLlama-13b-Instruct-hf'
#MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
EMB_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 2048
TEMP = 1e-10
MAX_INPUT_TOKEN_LENGTH = 6000

st.set_page_config(layout="wide")
st.title('Augur: Machine Query Translation')


st.markdown("""
<style>
    div[data-baseweb="textarea"] > div {
        height: 20vh !important;
    }
</style>
""", unsafe_allow_html=True)




# def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
#     input_token_length = get_input_token_length(message, chat_history, system_prompt)
#     if input_token_length > MAX_INPUT_TOKEN_LENGTH:
#         raise Exception(f'The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')
    

# def send_consult(query):
#     conn_details = {
#     'endpoint': 'https://sd-0d6bd678.stardog.cloud:5820',
#     'username': SD_USER,
#     'password': SD_PASSWORD
#     }

#     # Initialize connection
#     with stardog.Connection('PD_KG', **conn_details) as conn:        
#         # Execute the query
#         if 'CONSTRUCT' in query:
#             results = conn.graph(query)
#         else:
#             results = conn.select(query)

#     return results

@st.cache_resource()
def load_conversation_model():
    df_train = pd.read_json("datafiles/train-data.json")
    ontology_db = rag.GraphRag(EMB_MODEL_ID, ["./datafiles/dbpedia_2016-10.owl"])
    queries_db = rag.SparQLRag(EMB_MODEL_ID, 
                               df_train["corrected_question"].to_list(), 
                               df_train["sparql_query"].to_list())
    
    streamer, query_pipeline = model.conversational_pipeline_st(MODEL_ID, MAX_NEW_TOKENS)

    chat_code_agent = ctx.PromptLlamaCode(ontology_db, queries_db)

    return chat_code_agent, streamer, query_pipeline

# Load model and prompt formatter
code_agent, streamer, query_pipeline = load_conversation_model()

def capture_code(text): 
    code_pattern = r"```(?:sparql)?(.*?)```"
    code = re.findall(code_pattern, text, re.DOTALL)
    return code[0] if code else "None"


with st.sidebar:
    st.subheader('Model parameters')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=2.0, value=1., step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=1, max_value=100, value=10, step=5)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)

    st.subheader('In-context Learning')
    cove = st.checkbox('Chain of Verification')
    cot = st.checkbox('Chain of Thought')
    rag_ont = st.checkbox('Retrieval Augmented Generation')
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
               
                st.session_state.first_message = False

                conversation = model.conversation_init(code_agent, 
                                                   prompt, 
                                                   few_shot=few_shot,
                                                   cot=cot,
                                                   rag=rag_ont)

                if 'conversation' not in st.session_state:
                    st.session_state.conversation = conversation
                
                response = query_pipeline(
                    conversation,
                    do_sample=True,
                    top_k=1,
                    temperature=TEMP,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_return_sequences=1
                    )
    
                placeholder = st.empty()
                full_response = ''
                placeholder.markdown(response[-1]['content'])
                
                code = capture_code(response[2]['content'])

                # st.write('**Debug: Código capturado**')
                # code_box = st.empty()
                # st.session_state.output = code
                # output = f'```{st.session_state.output}```'
                # code_box.markdown(output)

                model_response = response[2]['content']

        message = {"role": "assistant", "content": model_response}
        st.session_state.messages.append(message)

with col2: 
    st.write("CAPTURED CONSULT")
    code_box = st.empty()
    code_box.markdown(f'```{st.session_state.output + "LIMIT 10"}')
    if st.button('SEND CONSULT'):
        consult_output = st.empty()
        json_string = ''#send_consult((st.session_state.output if 'output' in st.session_state else '') + 'LIMIT 10')
        try: 
            json_data = json.dumps(json_string, indent=4) 
            json_string = json_data
        except:
            print('Graph output')  
        
        consult_output.markdown(f"```json\n{json_string}\n```")
    else:
        st.write('Triplestore consult results')


