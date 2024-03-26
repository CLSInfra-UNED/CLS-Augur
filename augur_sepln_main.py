import re
import gc
import os 

import pandas as pd
import Augur.context_templates as ctx

from tqdm import tqdm
from torch.cuda import empty_cache

from Augur import (model, rag, utils, db_endpoint)
from Augur.agent_templates import StanzaTaggingAgent, GenerativeTaggingAgent

from torch import cuda
from transformers import Conversation

from rdflib import Graph
tqdm.pandas()

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
        conversation = model.conversation_init(
            chat_code_agent,
            row['corrected_question'],
            few_shot=True,
            cot=False,
            rag=False,
        )
        result = query_pipeline(
            conversation,
            do_sample=False,
            top_k=1,
            #temperature=TEMP,
            max_new_tokens=MAX_NEW_TOKENS
        )
        result_code = utils.capture_code(result.generated_responses[-1])
        prediction = preprocess(eval(db_endpoint.send_consult(result_code)))
        gold_std = preprocess(eval(db_endpoint.send_consult(row['sparql_query'])))
        success = prediction == gold_std
        validity = 0 if 'error' in prediction else 1
        error = 0
    except Exception as e:
        print(e)
        success = False
        validity = 0
        error = 1

    return pd.Series({
        'model': MODEL_NAME,
        'method': "agent_POS_FSRAG",
        'consult': result_code,
        'prompt': result[1]['content'],
        'validity': validity,
        'success': success,
        'error':error
    })

def process_query_consistency(row, chat_code_agent, query_pipeline, ontology_set):
    #def process_query_helper():

    try:
        conversation_list = model.conversation_init_dict(
            chat_code_agent,
            row['corrected_question'],
            few_shot=True,
            cot=False,
            rag=False,
        )
        conversation = Conversation()
        conversation.add_message(conversation_list[0])
        conversation.add_message(conversation_list[1])

        result = query_pipeline(
            conversation,
            do_sample=False,
            #top_k=1,
            #temperature=TEMP,
            max_new_tokens=MAX_NEW_TOKENS
        )
        result_code = utils.capture_code(result.generated_responses[-1])
        prediction = preprocess(eval(db_endpoint.send_consult(result_code)))
        gold_std = preprocess(eval(db_endpoint.send_consult(row['sparql_query'])))
        success = prediction == gold_std
        validity = 0 if 'error' in prediction else 1
        error = 0
    except Exception as e:
        print(e)
        success = False
        validity = 0
        error = 1

    # conversation = Conversation()
    # conversation.add_message(conversation_list[0])
    # conversation.add_message(conversation_list[1])
    
    # result_single = query_pipeline(
    #     conversation,
    #     do_sample=True,
    #     #top_k=20,
    #     temperature=0.9,
    #     max_new_tokens=MAX_NEW_TOKENS
    # )
    # _result_code = utils.capture_code(result_single.generated_responses[-1])
    # _prediction = preprocess(eval(db_endpoint.send_consult(_result_code)))
    # _success = _prediction == gold_std

    sc_outputs = []
    conversation_sc_list = []
    for _ in range(20):
        conversation = Conversation()
        conversation.add_message(conversation_list[0])
        conversation.add_message(conversation_list[1])
        conversation_sc_list.append(conversation)

    result_sc = query_pipeline(
        conversation_sc_list,
        do_sample=True,
        temperature=0.7,
        top_k=20,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=3
    )
    _result_code_sc = [utils.capture_code(conv_result.generated_responses[-1]) for conv_result in result_sc] 
    #sc_outputs.append(_result_code_sc)

    result_code_sc = utils.self_consistency(_result_code_sc, ontology_set)
    prediction_sc = preprocess(eval(db_endpoint.send_consult(result_code_sc)))
    success_sc = prediction_sc == gold_std
    print('sc',prediction_sc, gold_std, success_sc)
    validity_sc = 0 if 'error' in prediction_sc else 1

    global greedy_acc
    global sc_acc
    global sample_acc

    greedy_acc+=success
    sc_acc+=success_sc
    #sample_acc+=_success

    print(f"Greedy: {greedy_acc}")
    print(f"Self-C: {sc_acc}")
    print(f"Sample: {sample_acc}")

    return pd.Series({
        'model': MODEL_NAME,
        'method': "agent_SC",
        'consult': result_code,
        'prompt': conversation[1]['content'],
        'validity': validity,
        'success': success,
        'consult_sc': result_code_sc,
        'validity_sc': validity_sc,
        'success_sc': success_sc,
        #'success_sample' : _success
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

    dbpedia_graph = Graph().parse('./datafiles/dbpedia_2016-10_extended.owl', format='xml')

    ontology_set = {str(element) for triple in dbpedia_graph for element in triple if
            'http://dbpedia.org/' in element}

    #agent = StanzaTaggingAgent(model_id=MODEL_ID)

    chat_code_agent = ctx.PromptLlamaCode(
        schema_rag=ontology_db, 
        sparql_rag=queries_db, 
        #agent=agent
    )
    
    os.makedirs(f'{MODEL_NAME}sepln', exist_ok=True)

    # Instantiate conversational tools and prompt manager
    query_pipeline = model.conversational_pipeline(MODEL_ID, MAX_NEW_TOKENS)
    
    global greedy_acc
    global sc_acc
    global sample_acc
    greedy_acc = sc_acc = sample_acc = 0
 
    results_df = df_test.progress_apply(lambda x: process_query_consistency(
        row=x, 
        chat_code_agent=chat_code_agent, 
        query_pipeline=query_pipeline,
        ontology_set=ontology_set), axis=1)

    # Save to disk
    results_df.to_csv(f"{MODEL_NAME}sepln/{MODEL_NAME}_SC.csv")
    print(sum(results_df['success'].astype(int)))
    print(sum(results_df['validity'].astype(int)))
    print(sum(results_df['error'].astype(int)))

if __name__ == '__main__':
    main()