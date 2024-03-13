import stanza
import spacy_stanza
import spacy
import re
import json

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
        spacy.prefer_gpu()

        self.nlp = spacy_stanza.load_pipeline("en")
        self.pipe = conversational_pipeline(model_id=model_id)

    
    def get_definitions(self, tag_list):
        agg_output = dict()
        for element in tag_list:
            message = (f'Write a rdfs:comment suitable for the meaning of this'
                       f' dbpedia ontology identifier: {element}. Be succint.')
            convers = Conversation()
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
        aux_buffer = []
        for token in doc:
            if token.pos_ == "AUX":
                aux_buffer.append(token.text)
            elif token.pos_ == "VERB":
                if aux_buffer:
                    grouped_verb = " ".join(aux_buffer) + " " + token.text
                    verbs_with_aux.append(grouped_verb)
                    aux_buffer = []  # Clear
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


class GenerativeTaggingAgent(Agent):
    def __init__(self, model_id):
        self.pipe = conversational_pipeline(model_id=model_id)
        self.instruction = """
### EXAMPLE
# Write intermediate ontology form, extracting possible classes, properties, and relations, with generic descriptions: How many movies did Stanley Kubrick direct?
Format the answer in EXACTLY this json format:
```json
{
  "Film": "A Film represents a cinematic work or movie. It is a visual art form used to simulate experiences that communicate ideas, stories, perceptions, feelings, beauty, or atmosphere through the use of moving images. These images are often accompanied by sound, and more rarely, other sensory stimulations. The term 'film' encompasses both the medium and the form of expression.",
  "director": "The director of a film or movie is the individual responsible for overseeing the artistic and dramatic aspects, visualizing the screenplay, or script, while guiding the technical crew and actors in the fulfillment of that vision. The director has a key role in choosing the cast members, production design, and the creative aspects of filmmaking.",
  "Stanley_Kubrick": "Stanley Kubrick was an American film director, producer, screenwriter, and photographer. He is frequently cited as one of the greatest filmmakers in cinematic history. His films, which are mostly adaptations of novels or short stories, cover a wide range of genres, and are noted for their dark humor, unique cinematography, extensive set designs, and evocative use of music. Kubrick was known for his meticulous perfectionism, extensive research, and slow pace of filming."
}
```
### INSTRUCTION
"""
    def capture_json(self, text):
        code_pattern = r"```(?:json)?(.*?)```"
        code = re.findall(code_pattern, text, re.DOTALL)
        return code[0] if code else "None"
    
    def __call__(self, text):
        prompt = self.instruction + f"\n# Write intermediate ontology form, extracting possible classes, properties, and relations, with generic descriptions: {text}"
        convers = Conversation()
        convers.add_user_input(prompt)
        dialog = self.pipe(convers, max_new_tokens = 1000)
        output  = dialog.generated_responses[-1].strip('"')
        output = self.capture_json(output)
        output = json.loads(output)

        return output
        
