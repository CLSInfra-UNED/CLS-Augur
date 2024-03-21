import os 

from Augur import db_endpoint
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
tqdm.pandas()

import pandas as pd
import Augur.context_templates as ctx

from tqdm import tqdm

from Augur import (model, rag, utils, db_endpoint)
from Augur.agent_templates import StanzaTaggingAgent, OpenAITaggingAgent



MODEL_ID = 'codellama/CodeLlama-13b-Instruct-hf'
#MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
EMB_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 500
TEMP = 1e-10
MODEL_NAME = MODEL_ID.split('/')[-1]

def preprocess(output):
    _result = set()
    if 'head' in output and 'results' in output:
        subset = set()
        for element in output['results']['bindings']:
            subset |= ({element[x]['value'] for x in element.keys()})
        _result = subset
    elif 'head' in output and 'boolean' in output:
        _result.add(output['boolean'])
    else:
        print(output)
        _result.add('error')
    return _result

def process_query(row, chat_code_agent, query_pipeline):
    try:
        conversation = model.conversation_init_dict(
            chat_code_agent,
            row['corrected_question'],
            few_shot=True,
            cot=False,
            rag=False,
        )
        result = client.chat.completions.create(
            model = "gpt-3.5-turbo-0125",
            messages = conversation,
            temperature = 0,
        )
        result_code = utils.capture_code(result.choices[0].message.content)
        prediction = preprocess(eval(db_endpoint.send_consult(result_code)))
        gold_std = preprocess(eval(db_endpoint.send_consult(row['sparql_query'])))
        success = prediction == gold_std
        validity = 0 if 'error' in prediction else 1
    except Exception as e:
        print(e)
        success = False
        validity = 0

    return pd.Series({
        'model': MODEL_NAME,
        'method': "agent_GEN_FSRAG",
        'consult': result_code,
        'prompt': result[1]['content'],
        'validity': validity,
        'success': success,
    })


def main():
    # Load datasets
    df_test = pd.read_json("datafiles/test-data.json")
    df_test = df_test.sample(n=1000, random_state=42)
    df_train = pd.read_json("datafiles/train-data.json")

    ontology_files = [
        "./datafiles/dbpedia_2016-10_extended.owl",
    ]

    # Instantiate rag db objects
    ontology_db = rag.GraphRAG(EMB_MODEL_ID, ontology_files)
    queries_db = rag.SparQLRAG(
        EMB_MODEL_ID, 
        df_train["corrected_question"].to_list(), 
        df_train["sparql_query"].to_list()
    )

    agent = GenerativeTaggingAgent(model_id=MODEL_ID)

    chat_code_agent = ctx.PromptLlamaCode(
        schema_rag=ontology_db, 
        sparql_rag=queries_db, 
        agent=agent
    )
    
    os.makedirs(f'{MODEL_NAME}sepln', exist_ok=True)

    # Instantiate conversational tools and prompt manager
    query_pipeline = model.conversational_pipeline(MODEL_ID, MAX_NEW_TOKENS)

    results_df = df_test.progress_apply(lambda x: process_query(
        row=x, 
        chat_code_agent=chat_code_agent, 
        query_pipeline=query_pipeline), axis=1)

    # Save to disk
    results_df.to_csv(f"{MODEL_NAME}sepln/{MODEL_NAME}_AgentGen_FSRAG_k5.csv")
    print(sum(results_df['success'].astype(int)))
    print(sum(results_df['validity'].astype(int)))

if __name__ == '__main__':
    main()