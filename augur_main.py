import re
import gc
import os 

import pandas as pd
import Augur.context_templates as ctx

from torch.cuda import empty_cache

from Augur import (model, rag)
from datasets import load_dataset

MODEL_ID = 'codellama/CodeLlama-13b-Instruct-hf'
#MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
EMB_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 1024
TEMP = 1e-10

def capture_code(text): 
    code_pattern = r"```(?:sparql)?(.*?)```"
    code = re.findall(code_pattern, text, re.DOTALL)
    return code[0] if code else "None"

              
def main():
    df_test = pd.read_json("datafiles/test-data.json")
    df_train = pd.read_json("datafiles/train-data.json")

    ontology_files = [
        "./datafiles/dbpedia_2016-10.owl",
    ]

    ontology_db = rag.GraphRag(EMB_MODEL_ID, ontology_files)
    queries_db = rag.SparQLRag(EMB_MODEL_ID, 
                               df_train["corrected_question"].to_list(), 
                               df_train["sparql_query"].to_list())

    query_pipeline = model.conversational_pipeline(MODEL_ID, MAX_NEW_TOKENS)

    chat_code_agent = ctx.PromptLlamaCode(ontology_db, queries_db)
    
    model_name = MODEL_ID.split('/')[-1]
    os.makedirs(model_name, exist_ok=True)

    all_combined = []
    combinations = [(bool(i & 4), bool(i & 2), bool(i & 1)) for i in range(8)]
    for cot, few_s, rag_s in combinations:
        ordered_list = []
        for query in df_test["corrected_question"][:100]:

            conversation = model.conversation_init(chat_code_agent, 
                                                   query, 
                                                   few_shot=few_s,
                                                   cot=cot,
                                                   rag=rag_s)
            
            result  = query_pipeline(conversation,
                                    do_sample=True,
                                    #top_k=10,
                                    temperature=TEMP,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    num_return_sequences=1)
            
            result = {
                'model'  : model_name,
                'method' : f"{few_s * 'FS_'}{cot * 'CoT_'}{rag_s * 'ont_rag_'}",
                'consult' : capture_code(result[2]['content']),
                }
            ordered_list.append(result)
            all_combined.append(result)
            
            del conversation
            gc.collect()
            empty_cache()
        
        pd.DataFrame(ordered_list).to_csv((
            f"{model_name}/"
            f"{few_s * 'FS_'}"
            f"{cot * 'CoT_'}"
            f"{rag_s * 'ont_rag_'}"
            f"{model_name}.csv"
            )
        )

    pd.DataFrame(all_combined).to_csv(f"{model_name}/{model_name}_combined.csv")

    # conversation_list = []
    # query = df_test['corrected_question'][0]
    # #for query in df_test['corrected_question'][:100]:
    # for _ in range(100):
    #     conversation = model.conversation_init(chat_code_agent,
    #                                         query,
    #                                         few_shot = True,
    #                                         cot = False,
    #                                         rag = True)
        
    #     result =query_pipeline(conversation,
    #                 do_sample = True,
    #                 top_k = 10,
    #                 #temperature = TEMP,
    #                 max_new_tokens = MAX_NEW_TOKENS,
    #                 # num_return_sequences = 1,
    #                 #num_beams=1)
    #                 penalty_alpha= 0)
        
    #     conversation_list.append({'consult' : capture_code(result[2]['content'])})

    #     del result
    #     del conversation
    #     gc.collect()
    #     empty_cache()

    # pd.DataFrame(conversation_list).to_csv('deepseek_results_do_sample_true_topk1.csv')
    

if __name__ == '__main__':
    main()



