from Augur import db_endpoint
from rdflib import URIRef, Graph, RDFS, Literal, Namespace
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
    
    SCHEMA = '### Sample Identifiers from Database Schema.\n ### These identifiers are examples of what you might encounter in the database and are provided to assist in query construction. They represent a subset of the possible entities and relationships in the full schema:\n{schema}\n\n'
    #SCHEMA = "### Choose from this list those URIs that are helpful in building the SPARQL query requested:\n {schema}"
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
    COT_SEPLN = """
### Some example user requests and corresponding SparQL queries are provided based on similar problems to help you answer the last request:
# Write the SparQL code that retrieves the answer to this request: "Which architect of Marine Corps Air Station Kaneohe Bay was also a tenant of the New Sanno Hotel?"
For the question "Which architect of Marine Corps Air Station Kaneohe Bay was also a tenant of the New Sanno Hotel?", let's think step by step to derive an answer using ontology elements and SPARQL queries, with a focus on the relationships and entities involved.

1. Identify the Entities: The first step is recognizing the entities mentioned in the question. We have two main entities: "Marine Corps Air Station Kaneohe Bay" and "New Sanno Hotel". These entities are represented by their respective URIs in the ontology or database, indicating specific locations or institutions.
2. Understand the Relationships: Next, we need to comprehend the relationships or properties mentioned: "architect" and "tenant". These relationships connect our entities to the individual or individuals we're interested in finding. The "architect" relationship links a building or structure to its designer, while the "tenant" relationship associates a person or entity with a place where they reside or occupy space.
3. Formulate the SPARQL Query: With an understanding of the entities and relationships, we can construct a SPARQL query to find individuals that fulfill both roles—being the architect of the Marine Corps Air Station Kaneohe Bay and also a tenant of the New Sanno Hotel. The query is structured to select distinct URIs (?uri) that have the specified relationships with our entities.
4. Linking Entities to URIs: It's crucial to accurately link our entities to their corresponding URIs in the database. The URI for "Marine Corps Air Station Kaneohe Bay" is <http://dbpedia.org/resource/Marine_Corps_Air_Station_Kaneohe_Bay>, and for "New Sanno Hotel", it's <http://dbpedia.org/resource/New_Sanno_Hotel>. These URIs serve as exact references to our entities within the database, allowing us to retrieve specific information about them.
5. Mapping Relationships to Properties: Finally, we map the "architect" and "tenant" relationships to their respective properties in the ontology. The property <http://dbpedia.org/property/architect> connects buildings to their architects, and <http://dbpedia.org/ontology/tenant> connects locations to their tenants. By using these properties in our query, we can find the individual(s) who are linked to both entities through these specific relationships.
The SPARQL query that encapsulates this thought process is:

```sparql
SELECT DISTINCT ?uri 
WHERE { 
  <http://dbpedia.org/resource/Marine_Corps_Air_Station_Kaneohe_Bay> <http://dbpedia.org/property/architect> ?uri.
  <http://dbpedia.org/resource/New_Sanno_Hotel> <http://dbpedia.org/ontology/tenant> ?uri
}
```
This query seeks to find distinct individuals (?uri) who are both the architect of Marine Corps Air Station Kaneohe Bay and a tenant of the New Sanno Hotel by leveraging the specific relationships (properties) that connect these individuals to the mentioned entities.
"""
    COT_END = "Let's think step by step to derive an answer using ontology elements and SPARQL queries, with a focus on the relationships and entities involved."
    
    FEW_SHOT = (
        "### Some example user requests and corresponding SparQL queries are provided "
        "based on similar problems:\n"
    )
    FEW_SHOT_TEMPLATE = "# Write the SparQL code that retrieves the answer to this request: {question}\n{consult}\n"

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

    def search_resource(self, name_list):
        g = Graph()
        ids = set()
        for resource in name_list:
            query = (
                "SELECT ?s ?o  "
                "WHERE {"
                " ?s <http://www.w3.org/2000/01/rdf-schema#label> ?o ."
                f" ?o bif:contains '\"{resource}\"'@en"
                "} LIMIT 10" 
                )

            _output = db_endpoint.send_consult_json(query)
            if _output: _output = _output["results"]["bindings"]
            
            for result in _output:
                if 'Category:' in result["s"]["value"]: continue
                resource_uri = URIRef(result["s"]["value"])
                label = Literal(result["o"]["value"])
                g.add([resource_uri, RDFS.label, label])
                ids.add(str(resource_uri))
        
        # prefixes_to_unbind = [prefix for prefix, _ in g.namespace_manager.namespaces()]
        # for prefix in prefixes_to_unbind:
        #     g.namespace_manager.bind(prefix, None, replace=True)
        #return ' '.join([f'<{idss}> \n' for idss in ids])
        return g.serialize(format='turtle')
    
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
            _result = self.agent(user_query)
            _result_schema = self.schema_rag.process_query(_result['predicted_ids'], max_k=5)
            _result_schema= _result_schema + self.search_resource(_result['predicted_names'])
            
            prompt.append(
                self.SCHEMA.format(
                    schema=_result_schema
                )
            )
        elif cheating:
            prompt.append(
                self.SCHEMA.format(
                    schema=cheating,
                )
            )
        
        if self.agent and cheating:
            prompt.append(cheating)
        
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
        

        if chain_t: prompt.append(self.COT_SEPLN)

        prompt.append(f"# TASK:\n# Write the SparQL code that retrieves the answer to this request: {user_query}.\n\n")

        if chain_t: prompt.append(self.COT_END)

        return '\n'.join(prompt)
    