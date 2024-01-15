from enum import Enum
from abc import ABC, abstractmethod

class prompt_template(ABC):

    @abstractmethod
    def generate_prompt(self):
        ...


class prompt_llama_code(prompt_template):
    SYSTEM = ("You are a helpful, respectful and honest assistant expert coding"
              " in SparQL, ontologies and semantic web. Always answer the code in SparQL that retrieves the "
              "information asked. DO use regex for names in the inputs."
            )
    SCHEMA = '## Given the following turtle schema of an ontology:\n {schema}\n\n'
    COT = """
## Some example user requests and corresponding SparQL queries are provided based on similar problems:
## Answer the following: Give me the list of all the works from any author named teresa.
## Answer:
Let's think step by step to create a SPARQL query from a natural language request.

Step 1: Recognize Named Entities and Ontology Keys
- Identify key named entities and terms in the natural language request, such as 'poetic work', 'creator', or specific names.
- Match these entities with keys from the ontology: 'kos:' for creator roles (e.g., 'kos:Creator'), 'pdc:' for aspects of poetic works (e.g., 'pdc:PoeticWork', 'pdc:title', 'pdc:initiated', 'pdc:hasAgentRole', 'pdc:hasAgent', 'pdc:roleFunction', 'pdc:name', 'pdc:hasTimeSpan', 'pdc:date').

Step 2: Translate to Intermediate Form
- Map the recognized entities and terms to their corresponding ontology keys to create an intermediate representation.
- For instance, if the request mentions a creator named 'Teresa', map this to '?creator' and associate it with 'pdc:name' and 'kos:Creator' from the ontology.

Step 3: Generate SPARQL Code
- Construct the SPARQL query using the intermediate representation.
- Include necessary prefixes like 'PREFIX kos: <http://postdata.linhd.uned.es/kos/>', 'PREFIX pdc: <http://postdata.linhd.uned.es/ontology/postdata-core#>', 'PREFIX pdp: <http://postdata.linhd.uned.es/ontology/postdata-poeticAnalysis#>'.
- Formulate the 'SELECT' statement (e.g., 'SELECT ?work ?resultText ?creator ?date') and 'WHERE' clause based on the user's request and the intermediate form.
- Use conditions and variables that reflect the user's request, such as filtering for a specific creator with 'FILTER regex(?creator, "teresa", "i")'.

```
PREFIX kos: <http://postdata.linhd.uned.es/kos/>
PREFIX pdc: <http://postdata.linhd.uned.es/ontology/postdata-core#>
PREFIX pdp: <http://postdata.linhd.uned.es/ontology/postdata-poeticAnalysis#>

SELECT ?work ?resultText ?creator ?date
WHERE {
    ?work pdc:title ?resultText.
    ?work a pdc:PoeticWork.

    ?creation pdc:initiated ?work;  
    pdc:hasAgentRole ?ag.
    ?ag pdc:hasAgent ?person;
        pdc:roleFunction kos:Creator.
    ?person pdc:name ?creator.
    
    OPTIONAL {
        ?creation pdc:hasTimeSpan ?sp.
        ?sp pdc:date ?date.
    }

    FILTER regex(?creator, "teresa", "i")
}
```
"""
    FEW_SHOT = ("## Some example user requests and corresponding SparQL queries are provided "
                "based on similar problems:\n"
                "## Answer the following: {question}\n\n ## Answer:\n{query}\n\n"
            )

    COT_END = "Let's think step by step."

    def __init__(self, schema_rag, sparql_rag):
       self.schema_rag = schema_rag
       self.sparql_rag = sparql_rag

    def generate_prompt(self, user_query, few_s = False, chain_t = False, rag = False):
        prompt = []

        if rag: prompt.append(self.SCHEMA.format(schema=self.schema_rag))
        if few_s: prompt.append(self.FEW_SHOT)
        if chain_t: prompt.append(self.COT)

        prompt.append(f"## Answer the following: {user_query}.\n\n## Answer:\n")

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