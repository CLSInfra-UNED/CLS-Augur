from enum import Enum
from collections import namedtuple
from abc import ABC, abstractmethod

class PromptTemplate(ABC):

    @abstractmethod
    def generate_prompt(self):
        ...


class PromptLlamaCode(PromptTemplate):
    SYSTEM = ("You are a helpful, respectful and honest assistant expert coding"
              " in SparQL, ontologies and semantic web. Always DO answer the code in SparQL that retrieves the "
              "information asked. DO enclose the code in a code block:\n```\ncode\n```\n).\n"
            )
    
    SCHEMA = '## Given the following turtle schema of an ontology:\n{schema}\n\n'
    
    COT = """
### Some example user requests and corresponding SparQL queries are provided based on similar problems to help you answer the last request:
# Write the SparQL code that retrieves the answer to this request: How many movies did Stanley Kubrick direct?

Let's think step by step, to create a SPARQL query from a natural language request:

Step 1: Recognize Named Entities and Ontology Entities
- Identify key named entities and terms in the natural language request. The key named entities in the request are "movies" and "Stanley Kubrick".
- Match these entities with entities from the ontology: These entities correlate with ontology entities like 'dbo:' for DBpedia ontology (e.g., 'dbo:Film', 'dbo:director') and 'dbr:' for DBpedia resources (e.g., 'dbr:Stanley_Kubrick').

Step 2: Translate to Intermediate Form
- Map the recognized entities and terms to their corresponding ontology entities from the GIVEN schema to create an intermediate representation.
- Map "movies" to the DBpedia ontology entity '?film' associated with 'dbo:Film'. 
- Map the entities found to their matching ontology resource: "Stanley Kubrick" is a DBpedia resource, represented as 'http://dbpedia.org/resource/Stanley_Kubrick'.

Step 3: Generate SPARQL Code
- Construct the SPARQL query using the intermediate representation.
- Include necessary prefixes like 'PREFIX dbo: http://dbpedia.org/ontology/' and 'PREFIX dbr: http://dbpedia.org/resource/'.
- Formulate the 'SELECT' statement to count the number of films: 'SELECT DISTINCT COUNT(?film)', with COUNT clause based on the user's request and the intermediate form.
- Construct the 'WHERE' clause to specify the films directed by Stanley Kubrick: 'WHERE {?film dbo:director dbr:Stanley_Kubrick . }'. If it's a name, use regex for names so they can be separated by underscore or spaces.

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT DISTINCT COUNT(?film)
WHERE {
  ?film dbo:director dbr:Stanley_Kubrick .
}
```
"""
    COT_END = "Let's think step by step,"
    
    FEW_SHOT = (
        "### Some example user requests and corresponding SparQL queries are provided "
        "based on similar problems:\n"
    )
    FEW_SHOT_TEMPLATE = "# Write the SparQL code that retrieves the answer to this request: {question}\n\n{consult}\n<|EOT|>\n\n"

    
    def __init__(self, schema_rag, sparql_rag):
       self.schema_rag = schema_rag
       self.sparql_rag = sparql_rag

    def generate_prompt(self, user_query, few_s = False, chain_t = False, rag = False):
        prompt = []

        if rag: prompt.append(self.SCHEMA.format(schema = self.schema_rag.process_query(user_query), k = 5))
       
        if few_s:
            fs_rag_output = self.sparql_rag.process_query(user_query)
            
            few_shot_concat = [
                self.FEW_SHOT_TEMPLATE.format(
                    question = document['question'],
                    consult = "```sparql\n" + document['metadata']['consult'] + "\n```"
                    )
                    for document in fs_rag_output
                ]
            
            few_shot_concat = '\n'.join(few_shot_concat)
            prompt.append(self.FEW_SHOT + few_shot_concat)
        
        if chain_t: prompt.append(self.COT)

        prompt.append(f"# Write the SparQL code that retrieves the answer to ONLY this request: {user_query}.\n\n")

        if chain_t: prompt.append(self.COT_END)

        return '\n'.join(prompt)
    





COT_EXAMPLE = """
Natural Language Query: "Find articles written by John Doe about Artificial Intelligence."
Chain of Thought Reasoning:
1. Identify the author: John Doe.
2. Identify the subject: Artificial Intelligence.
3. Search for articles matching these criteria.
SPARQL Query: 
"SELECT ?article WHERE { ?article wdt:P50 wd:JohnDoe . ?article wdt:P921 wd:ArtificialIntelligence . }"
"""