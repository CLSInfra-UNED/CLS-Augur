import re
import Augur.context_templates as ctx

from Augur import (model, rag)
from datasets import load_dataset

MODEL_NAME = 'codellama/CodeLlama-13b-Instruct-hf'
EMB_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 1024
TEMP = .01

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
    
    query = 'test'
    conversation = model.conversation_init(chat_code_agent, query)
    
    code_pattern = r'```(.*?)```'
    code = re.findall(code_pattern, query, re.DOTALL)
    
    ordered_list.append(query_pipeline(conversation,
                            do_sample=True,
                            top_k=10,
                            temperature=TEMP,
                            max_new_tokens=MAX_NEW_TOKENS,
                            num_return_sequences=1))

if __name__ == '__main__':
    main()



