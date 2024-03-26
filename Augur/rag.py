import datetime
import math

from abc import ABC, abstractmethod

from functools import lru_cache
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rdflib import Graph
from rdflib.namespace import RDFS


@lru_cache(maxsize=1)
def load_embedding_model(emb_model):
    return HuggingFaceEmbeddings(model_name=emb_model)


class RAGBase(ABC):
    def __init__(self, emb_model_id):
        self.emb_model = load_embedding_model(emb_model_id)
        self.unique_db_dir = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @abstractmethod
    def process_query(self):
        ...

    @abstractmethod
    def load_vector_db(self):
        ...

    @abstractmethod
    def raw_rag_output(self):
        ...


class GraphRAG(RAGBase):
    """
    Class intended to manage the vector db for a Knowled Graph
    """

    def __init__(self, emb_model_id, file_list):
        super().__init__(emb_model_id)
        self.graph = Graph()
        self.unique_db_dir = 'db_ontology/' + self.unique_db_dir
        self.rag = self.load_vector_db(file_list)


    def load_vector_db(self, file_list):
        # Iterates over the list of file names and parse them
        # based on the file extension
        for file in file_list:
            file_format = file.split('.')[-1]
            match file_format:
                case 'ttl': self.graph.parse(file, format='ttl')
                case 'owl': self.graph.parse(file, format='xml')
                case _: print('Unrecognized format: Skipping file.')

        # Iterates over a generator that filters the graph saving
        # all tripletswith a 'comment' predicate
        filtered_graph = (
            (subj, pred, obj) for subj, pred, obj in self.graph
            if pred == RDFS.comment and (obj.language == 'en' or not obj.language)
        )
        node_list, metadata = [], []
        for subj, pred, obj in filtered_graph:
            node_list.append(obj)
            metadata.append({'subj': subj, 'pred': pred, 'obj': obj})

        # Loads the triplets into the vector database
        db = Chroma.from_texts(
            node_list,
            embedding=self.emb_model,
            metadatas=metadata,
            persist_directory=self.unique_db_dir
        )

        return db
    

    def raw_rag_output(self, text, k=10):
        return self.rag.similarity_search(text, k=k)
    

    def _get_connected_nodes_and_prefixes(self, node):
        connected_nodes = set()
        for subj, pred, obj in self.graph:
            if str(subj) in node:
                connected_nodes.add(subj)
                connected_nodes.add(obj)
            # if str(obj) in node:
            #     connected_nodes.add(subj)
        connected_graph = Graph()
        for subj, pred, obj in self.graph:
            if subj in connected_nodes:
                if (pred == RDFS.label and obj.language != 'en' or
                        pred == RDFS.comment and (obj.language and obj.language != 'en')):
                    continue
                connected_graph.add((subj, pred, obj))

        # for prefix, namespace in self.graph.namespaces():
        #     connected_graph.bind(prefix, namespace)

        return connected_graph
    

    def process_query(self, text, max_k=10):
        if isinstance(text, str): text = [text]

        k = max(1,max_k//len(text))
        k = math.ceil(max_k/len(text))

        output = [self.raw_rag_output(definition, k=k) for definition in text]
        nodes = [{document.metadata['subj'] for document in retrieved_output} for retrieved_output in output]
        nodes = set.union(*nodes)
        output = self._get_connected_nodes_and_prefixes(nodes)
        output = output.serialize(format='turtle')
        # Add dbr prefix, as the OWL file does not contain the resource triples
        #output = "@prefix dbr: <http://dbpedia.org/resource/> .\n" + output
        return output
    

    def full_schema(self):
        full_graph = Graph()
        for subj, pred, obj in self.graph:
            if (pred == RDFS.label and obj.language != 'en' or
                    pred == RDFS.comment and obj.language != 'en'):
                continue
            if ('#Class' in str(obj) or
                '#Property' in str(obj) or
                '#domain' in str(pred) or
                    '#range' in str(pred)):
                full_graph.add((subj, pred, obj))

        for prefix, namespace in self.graph.namespaces():
            full_graph.bind(prefix, namespace)

        full_graph = full_graph.serialize(format='turtle')

        return "@prefix dbr: <http://dbpedia.org/resource/> .\n" + full_graph


class SparQLRAG(RAGBase):
    """
    This class is intended to implement the RAG to retrieve
    relevant samples for in-context learning.
    """

    def __init__(self, emb_model_id, queries, consults):
        super().__init__(emb_model_id)
        self.unique_db_dir = 'db_sparql/' + self.unique_db_dir
        self.rag = self.load_vector_db(queries, consults)
        

    def load_vector_db(self, queries, consults):
        consults = [{'consult': element} for element in consults]
        db = Chroma.from_texts(
            queries,
            embedding=self.emb_model,
            metadatas=consults,
            persist_directory=self.unique_db_dir
        )
        return db


    def process_query(self, text, k=8):
        output = self.raw_rag_output(text, k)
        output = [{'question': document.page_content, 'metadata': document.metadata}
                  for document in output]
        return output

    
    def raw_rag_output(self, text, k=8):
        return self.rag.similarity_search(text, k=k)


if __name__ == "__main__":
    # TEST CASE
    import pandas as pd

    ontology_files = [
        './datafiles/dbpedia_2016-10_extended.owl'
    ]

    model = "sentence-transformers/all-mpnet-base-v2"

    df_train = pd.read_json('datafiles/train-data.json')

    ontology_db = GraphRAG(model, ontology_files)
    # fewshot_db = SparQLRAG(model,
    #                        df_train['corrected_question'].to_list(),
    #                        df_train['sparql_query'].to_list())

    query = " Who is the stockholder of the road tunnels operated by the Massachusetts Department of Transportation?"
    print(ontology_db.process_query(query, max_k=10))

    #print(fewshot_db.process_query(query, 5))
    print('done')
