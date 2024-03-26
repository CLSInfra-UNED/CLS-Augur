from enum import Enum
from collections import namedtuple
from abc import ABC, abstractmethod

class PromptTemplate(ABC):

    @abstractmethod
    def generate_prompt(self):
        ...


class PromptLlamaCode(PromptTemplate):
    SYSTEM = (
        "You are a helpful, respectful and honest assistant expert coding"
        " in SparQL, ontologies and semantic web. Always DO answer the code in SparQL that retrieves the "
        "information asked. DO enclose the code in a code block:\n```\ncode\n```\n).\n"
    )
    
    SCHEMA = '###Sample Identifiers from Database Schema.\n # These identifiers are examples of what you might encounter in the database and are provided to assist in query construction. They represent a subset of the possible entities and relationships in the full schema:\n{schema}\n\n'
    
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
- Construct the 'WHERE' clause to specify the films directed by Stanley Kubrick: 'WHERE {?film dbo:director dbr:Stanley_Kubrick . }'.

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

    FIXED_FS = """
### Some example user requests and corresponding SparQL queries are provided based on similar problems:
# Write the SparQL code that retrieves the answer to this request: How many movies did Stanley Kubrick direct?\n
```sparql
SELECT DISTINCT COUNT(?uri) WHERE {?uri <http://dbpedia.org/ontology/director> <http://dbpedia.org/resource/Stanley_Kubrick>  . }
```
<|EOT|>

# Write the SparQL code that retrieves the answer to this request: Which city's foundeer is John Forbes?
```sparql
SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/ontology/founder> <http://dbpedia.org/resource/John_Forbes_(British_Army_officer)>  . ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/City>}
```
<|EOT|>

# Write the SparQL code that retrieves the answer to this request: What is the river whose mouth is in deadsea?
```sparql
SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/ontology/riverMouth> <http://dbpedia.org/resource/Dead_Sea>  . ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/River>}
```
<|EOT|>

# Write the SparQL code that retrieves the answer to this request: What is the allegiance of John Kotelawala ?
```sparql
SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/John_Kotelawala> <http://dbpedia.org/property/allegiance> ?uri }
```
<|EOT|>

# Write the SparQL code that retrieves the answer to this request: How many races have the horses bred by Jacques Van't Hart participated in?
```sparql
SELECT DISTINCT COUNT(?uri) WHERE { ?x <http://dbpedia.org/ontology/breeder> <http://dbpedia.org/resource/Jacques_Van't_Hart> . ?x <http://dbpedia.org/property/race> ?uri  . }
```
<|EOT|>

# Write the SparQL code that retrieves the answer to this request: What is the incumbent of the Al Gore presidential campaign, 2000 and also the president of the Ann Lewis ?
```sparql
SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Al_Gore_presidential_campaign,_2000> <http://dbpedia.org/ontology/incumbent> ?uri. <http://dbpedia.org/resource/Ann_Lewis> <http://dbpedia.org/ontology/president> ?uri} 
```
<|EOT|>

# Write the SparQL code that retrieves the answer to this request: Was Ganymede discovered by Galileo Galilei?
```sparql
ASK WHERE { <http://dbpedia.org/resource/Ganymede_(moon)> <http://dbpedia.org/property/discoverer> <http://dbpedia.org/resource/Galileo_Galilei> }
```
<|EOT|>

# Write the SparQL code that retrieves the answer to this request: Does the Toyota Verossa have the front engine design platform?
```sparql
ASK WHERE { <http://dbpedia.org/resource/Toyota_Verossa> <http://dbpedia.org/ontology/automobilePlatform> <http://dbpedia.org/resource/Front-engine_design>  . }
```
<|EOT|> 
"""


    def __init__(self, schema_rag, sparql_rag, agent=None):
       self.schema_rag = schema_rag
       self.sparql_rag = sparql_rag
       self.agent = agent

    
    def generate_prompt(
            self,
            user_query,
            few_s=False,
            chain_t=False,
            rag=False,
            cheating=False
    ):
        prompt = []

        if rag: 
            prompt.append(
                self.SCHEMA.format(
                    schema=self.schema_rag.process_query(user_query),
                    max_k=5
                )
            )
        elif self.agent:
            prompt.append(
                self.SCHEMA.format(
                    schema=self.schema_rag.process_query(self.agent(user_query), max_k=5),
                )
            )
        elif cheating:
            prompt.append(
                self.SCHEMA.format(
                    schema=cheating,
                )
            )
        
        if few_s:
            fs_rag_output = self.sparql_rag.process_query(user_query)
            
            few_shot_concat = [
                (
                    # self.SCHEMA.format(
                    #     schema=self.schema_rag.process_query(
                    #         self.agent(document['question']), max_k=1) if self.agent else 
                    #         document['question'],
                    # ) +
                    self.FEW_SHOT_TEMPLATE.format(
                        question=document['question'],
                        consult="```sparql\n" +
                        document['metadata']['consult'] + "\n```"
                    )
                )
                for document in fs_rag_output
            ]
                
            
            few_shot_concat = '\n'.join(few_shot_concat)
            prompt.append(self.FEW_SHOT + few_shot_concat)

            # prompt.append(self.FIXED_FS)
        if chain_t: prompt.append(self.COT)

        prompt.append(f"# TASK:\n# Write the SparQL code that retrieves the answer to this request: {user_query}.\n\n")

        if chain_t: prompt.append(self.COT_END)

        return '\n'.join(prompt)
    