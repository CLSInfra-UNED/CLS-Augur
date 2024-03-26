import stanza
import spacy_stanza
import spacy
import re
import json
import os

from Augur import db_endpoint
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

from abc import ABC, abstractmethod
from transformers import Conversation
from .model import load_quant_model, conversational_pipeline

class Agent(ABC):
    @abstractmethod
    def __call__(self):
        pass

class StanzaTaggingAgent(Agent):
    def __init__(self, model_id):
        stanza.download("en")
        spacy.require_cpu()
        self.nlp = spacy_stanza.load_pipeline("en", use_gpu=False)
        self.pipe = conversational_pipeline(model_id=model_id)

    
    def get_definitions(self, tag_list):
        agg_output = dict()
        for element in tag_list:
            message = (f"Write a rdfs:comment suitable for the meaning of this"
                       f" dbpedia ontology identifier: {element}. Be succint.")
            convers = Conversation()
            convers.add_message({'role':'system', 'content':("###You are a helpful, respectful and honest assistant expert coding,"
                                                            " ontologies and semantic web. ONLY give the answer to the task."
                                                            "\n# EXAMPLE: Aviation\n Aviation the activity of flying aircraft, or of designing, producing, and keeping them in good condition.")})
            convers.add_user_input(message)
            dialog = self.pipe(convers, max_new_tokens = 1000)
            output  = dialog.generated_responses[-1].strip('"')
            agg_output[element] = output
        return agg_output
    

    def __call__(self, text):
        doc = self.nlp(text)

        verbs_with_aux = []
        propn_groups = []
        nouns = []
        pron_adv = []
        aux_buffer = []

        for token in doc:
            if token.pos_ in {'PRON', 'ADV'} and 'PronType=Int' in token.morph: 
                pron_adv.append(token.text)
            if token.pos_ == "AUX":
                aux_buffer.append(token.text)
            elif token.pos_ == "VERB":
                if aux_buffer:
                    grouped_verb = " ".join(aux_buffer) + " " + token.text
                    verbs_with_aux.append(grouped_verb)
                    aux_buffer = [] 
                else:
                    verbs_with_aux.append(token.text)
            elif token.pos_ == "PROPN":
                if propn_groups and token.i - 1 == propn_groups[-1][-1].i:
                    propn_groups[-1].append(token)
                else:
                    propn_groups.append([token])
            elif token.pos_ == "NOUN":
                nouns.append(token.text)
            else:
                aux_buffer = []

        #Merge
        propn_groups_merged = [" ".join([token.text for token in group]) for group in propn_groups]

        final_list = verbs_with_aux + propn_groups_merged + nouns
        output = self.get_definitions(final_list)

        return output

class OpenAIStanzaTaggingAgent(StanzaTaggingAgent):
    def __init__(self):
        stanza.download("en")
        spacy.require_cpu()
        self.nlp = spacy_stanza.load_pipeline("en", use_gpu=False)
    
    def get_definitions(self, tag_list):
        agg_output = dict()
        for element in tag_list:
            conversation = [
                {
                    'role': 'system',
                    'content': (
                        "###You are a helpful, respectful and honest assistant expert coding,"
                        " ontologies and semantic web. ONLY give the answer to the task."
                        "\n# EXAMPLE: Aviation\n Aviation the activity of flying aircraft, "
                        "or of designing, producing, and keeping them in good condition."
                    )
                },
                {
                    'role': 'user',
                    'content': (
                        f"Write a rdfs:comment suitable for the meaning of this"
                        f" dbpedia ontology identifier: {element}. Be succint."
                    )
                }
            ]
            result = client.chat.completions.create(
                    model = "gpt-3.5-turbo-0125",
                    messages = conversation,
                    temperature = 0,
                )
            output  = result.choices[0].message.content.strip('"')
            agg_output[element] = output
        return agg_output
    

class GenerativeTaggingAgent(Agent):
    def __init__(self, model_id):
        self.pipe = conversational_pipeline(model_id=model_id)
        self.instruction = """
###You are a helpful, respectful and honest assistant expert coding,  ontologies and semantic web. ONLY perform the instruction requested, in a single JSON. DO enclose the code in a code block:
```json
code
```

### EXAMPLE
# Extract possible classes, properties, and relations. Write a JSON with the names and succint generic descriptions of them: How many movies did Stanley Kubrick direct?
DO Format the answer in EXACTLY this json format:
```json
{
  "Film": "A Film represents a cinematic work or movie. It is a visual art form used to simulate experiences that communicate ideas, stories, perceptions, feelings, beauty, or atmosphere through the use of moving images.",
  "director": "The director of a film or movie is the individual responsible for overseeing the artistic and dramatic aspects, visualizing the screenplay, or script, while guiding the technical crew and actors in the fulfillment of that vision.",
  "Stanley_Kubrick": "Stanley Kubrick was an American film director, producer, screenwriter, and photographer. He is frequently cited as one of the greatest filmmakers in cinematic history."
}
```


### INSTRUCTION
"""
    def extract_first_json_object(self,text):
        open_braces = 0
        first_object = ""
        started = False

        for char in text:
            if char == "{":
                open_braces += 1
                started = True
            elif char == "}":
                open_braces -= 1
            
            if started:
                first_object += char
            
            if open_braces == 0 and started:
                break

        return first_object
    
    def capture_json(self, text):
        code_pattern = r"```(?:json)?(.*?)```"
        code = re.findall(code_pattern, text, re.DOTALL)
        return code[0] if code else "None"
    
    def __call__(self, text):
        prompt = self.instruction + f"\n# Extract possible classes, properties, and relations. Write a JSON with the names and succint generic descriptions of them: {text}." 
        prompt = prompt + """
DO use the following format:
```json
{
    string:string,
    string:string,
 }
```
DO NOT use nested fields. DO NOT forget the ',' after each element.
"""
        convers = Conversation()
        convers.add_message({'role':'user', 'content': prompt})
        dialog = self.pipe(convers, max_new_tokens = 1000, temperature=0.5)
        output  = dialog.generated_responses[-1].strip('"')
        output = self.capture_json(output)
        output = self.extract_first_json_object(output)
        try:
            output = json.loads(output)
        except Exception as e:
            print(e, output)
            output = {'default':'default'}

        return output

class OpenAITaggingAgent(Agent):
    def __init__(self):
        self.instruction = """
###You are a helpful, respectful and honest assistant expert coding,  ontologies and semantic web. ONLY perform the instruction requested, in a single JSON. DO enclose the code in a code block:
```json
code
```

### EXAMPLE
# Extract all possible classes, properties, and relations. Write a JSON with the names and succint generic descriptions of them: How many movies did Stanley Kubrick direct?
DO Format the answer in EXACTLY this json format:
```json
{
  "Film": "A Film represents a cinematic work or movie. It is a visual art form used to simulate experiences that communicate ideas, stories, perceptions, feelings, beauty, or atmosphere through the use of moving images.",
  "director": "The director of a film or movie is the individual responsible for overseeing the artistic and dramatic aspects, visualizing the screenplay, or script, while guiding the technical crew and actors in the fulfillment of that vision.",
  "Stanley_Kubrick": "Stanley Kubrick was an American film director, producer, screenwriter, and photographer. He is frequently cited as one of the greatest filmmakers in cinematic history."
}
```


### INSTRUCTION
"""
    def extract_first_json_object(self,text):
        open_braces = 0
        first_object = ""
        started = False

        for char in text:
            if char == "{":
                open_braces += 1
                started = True
            elif char == "}":
                open_braces -= 1
            
            if started:
                first_object += char
            
            if open_braces == 0 and started:
                break

        return first_object
    
    def capture_json(self, text):
        code_pattern = r"```(?:json)?(.*?)```"
        code = re.findall(code_pattern, text, re.DOTALL)
        return code[0] if code else "None"

    def __call__(self, text):
        prompt = self.instruction + f"\n# Extract all possible classes, properties, and relations. Write a JSON with the names and succint generic descriptions of them: {text}."
        prompt = prompt + """
DO use the following format:
```json
{
    string:string,
    string:string,
 }
```
DO NOT use nested fields. DO NOT forget the ',' after each element.
"""
        conversation = {"role": "user", "content": prompt},
        result = client.chat.completions.create(
                    model = "gpt-3.5-turbo-0125",
                    messages = conversation,
                    temperature = 0,
                )

        output = self.capture_json(result.choices[0].message.content)
        output = self.extract_first_json_object(output)
        try:
            output = json.loads(output)
        except Exception as e:
            print(e, output)
            output = {'default':'default'}

        return output


class OpenAIPreFormatTagging(OpenAITaggingAgent):
    def __init__(self):
        super().__init__(self)
    
    def __call__(self):
        pass

