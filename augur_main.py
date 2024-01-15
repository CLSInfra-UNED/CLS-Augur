import re
import Augur.context_templates as ctx

from Augur import (model, rag)
from datasets import load_dataset

MODEL_NAME = 'codellama/CodeLlama-13b-Instruct-hf'
EMB_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 1024
TEMP = .01

def capture_code(text): 
    code_pattern = r'```(.*?)```'
    code = re.findall(code_pattern, text, re.DOTALL)
    return code

              
def main():
    dataset  = load_dataset('lc_quad')

    ontology_files = [
        # "./datafiles/ontology.ttl",
        # "./datafiles/ontology_poetry.ttl",
        "./datafiles/dbpedia_2016-10.owl"
    ]

    ontology_db = rag.GraphRag(EMB_MODEL_ID, ontology_files)
    queries_db = rag.SparQLRag(EMB_MODEL_ID, ontology_files)

    query_pipeline = model.conversational_pipeline(MODEL_NAME, MAX_NEW_TOKENS)

    chat_code_agent = ctx.prompt_llama_code(ontology_db, queries_db)
    ordered_list = []

    
    combinations = [(bool(i & 4), bool(i & 2), bool(i & 1)) for i in range(8)]
    for cot, few_s, rag_s in combinations:
        for query in dataset['train']['question'][:100]:

            conversation = model.conversation_init(chat_code_agent, 
                                                query, 
                                                few_shot=few_s,
                                                cot=cot,
                                                rag=rag_s)
            
            ordered_list.append(query_pipeline(conversation,
                                    do_sample=True,
                                    top_k=10,
                                    temperature=TEMP,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    num_return_sequences=1))

if __name__ == '__main__':
    main()



