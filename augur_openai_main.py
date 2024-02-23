import re
import gc
import os

import pandas as pd
import Augur.context_templates as ctx

from dotenv import load_dotenv
from torch.cuda import empty_cache
from openai import OpenAI

from Augur import (model, rag)

load_dotenv()

OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
EMB_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 2048
TEMP = 0


def capture_code(text):
    code_pattern = r"```(?:sparql)?(.*?)```"
    code = re.findall(code_pattern, text, re.DOTALL)
    return code[0] if code else "None"


def main():
    # Load datasets
    df_test = pd.read_json("datafiles/test-data.json")
    df_train = pd.read_json("datafiles/train-data.json")

    ontology_files = [
        "./datafiles/dbpedia_2016-10.owl",
    ]

    # Instantiate rag db objects
    ontology_db = rag.GraphRag(EMB_MODEL_ID, ontology_files)
    queries_db = rag.SparQLRag(EMB_MODEL_ID,
                               df_train["corrected_question"].to_list(),
                               df_train["sparql_query"].to_list())

    # Instantiate conversational tools and prompt manager

    chat_code_agent = ctx.PromptLlamaCode(ontology_db, queries_db)

    model_name = "gpt-3.5-turbo"
    os.makedirs(model_name, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    #Perform inference with all method combinations
    all_combined = []
    combinations = [(bool(i & 4), bool(i & 2), bool(i & 1)) for i in range(8)]
    for cot, few_s, rag_s in combinations:
        ordered_list = []
        for query in df_test["corrected_question"][:100]:
            
            conversation = model.conversation_init_dict(chat_code_agent,
                                                   query,
                                                   few_shot = few_s,
                                                   cot = cot,
                                                   rag = rag_s)

            result = client.chat.completions.create(
                model = model_name,
                messages = conversation,
                temperature = 0,
                )
            result = result.choices[0].message.content

            result = {
                'model'  : model_name,
                'method' : f"{few_s * 'FS'}{cot * 'CoT'}{rag_s * 'ont_rag'}",
                'consult' : capture_code(result),
                }

            ordered_list.append(result)
            all_combined.append(result)

            # Clear memory cache
            del conversation
            gc.collect()
            empty_cache()

        # Save to disk partial result
        pd.DataFrame(ordered_list).to_csv((
            f"{model_name}/"
            f"{few_s * 'FS_'}"
            f"{cot * 'CoT_'}"
            f"{rag_s * 'ont_rag_'}"
            f"{model_name}.csv"
            )
        )

    # Save to disk combined results
    pd.DataFrame(all_combined).to_csv(f"{model_name}/{model_name}_combined.csv")

    # conversation_list = []
    # for query in df_test['corrected_question'][:100]:
    #     conversation = model.conversation_init(chat_code_agent,
    #                                         query,
    #                                         few_shot = True,
    #                                         cot = False,
    #                                         rag = True)

    #     result =query_pipeline(conversation,
    #                 do_sample = True,
    #                 top_k = 1,
    #                 temperature = TEMP,
    #                 max_new_tokens = MAX_NEW_TOKENS,)
    #                 # num_return_sequences = 1,
    #                 #num_beams=1)
    #                 #penalty_alpha= 0)

    #     conversation_list.append({'consult' : capture_code(result[2]['content'])})

    #     del result
    #     del conversation
    #     gc.collect()
    #     empty_cache()

    # pd.DataFrame(conversation_list).to_csv('deepseek_results_fs8_CR_response.csv')


if __name__ == '__main__':
    main()
