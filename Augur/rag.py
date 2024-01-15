import datetime 

from functools import lru_cache
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from rdflib import Graph
from abc import ABC, abstractmethod

@lru_cache(maxsize=1)
def load_embedding_model(emb_model):
    return HuggingFaceEmbeddings(model_name=emb_model)#, model_kwargs=model_kwargs)


class RagBase(ABC):
    def __init__(self, emb_model_id):
        self.emb_model = load_embedding_model(emb_model_id)
        self.unique_db_dir = f"../db/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


    @abstractmethod
    def process_query(self):
        ...


    @abstractmethod
    def load_vector_db(self):
        ...


    @abstractmethod
    def raw_rag_output(self):
        ...


class GraphRag(RagBase):

    def __init__(self, emb_model_id, file_list):
        super().__init__(emb_model_id)
        self.graph = Graph()
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
            if 'comment' in pred.lower() #or 'label' in pred.lower()
            )
        node_list, metadata = [], []
        for subj, pred, obj in filtered_graph:
            node_list.append(obj)
            metadata.append({'subj': subj, 'pred': pred, 'obj': obj})

        # Loads the triplets into the vector database
        db = Chroma.from_texts(node_list,
                            embedding = self.emb_model,
                            metadatas = metadata)
        
        return db
    

    def raw_rag_output(self, text, k = 10):
        return self.rag.similarity_search(text, include_metadata=True, k = k)


    def _get_connected_nodes_and_prefixes(self, node):
        
        connected_nodes = set()
        for subj, pred, obj in self.graph:
            if str(subj) in node:
                connected_nodes.add(subj)
                connected_nodes.add(obj)

        connected_graph = Graph()
        for subj, pred, obj in self.graph:
            if subj in connected_nodes: connected_graph.add((subj, pred, obj))

        for prefix, namespace in self.graph.namespaces(): 
            connected_graph.bind(prefix, namespace) 

        return connected_graph
    

    def process_query(self, text, k = 10):
        output = self.raw_rag_output(text, k)
        nodes = {document.metadata['subj'] for document in output}
        output = self._get_connected_nodes_and_prefixes(nodes)
        output = output.serialize(format = 'turtle')

        return output


class SparQLRag(RagBase):
    """
    WIP. This class is intended to implement the RAG to retrieve
    relevant samples for in-context learning.
    """
    def __init__(self, emb_model_id, file_list):
        super().__init__(emb_model_id)

    def load_vector_db(emb_model, file_list):
        #model_kwargs = {'device': 'cpu'}
        pass
    
    def process_query():
        pass

    def raw_rag_output():
        pass


if __name__ == "__main__":
    # TEST CASE
    ontology_files = [
    #"./datafiles/ontology.ttl",
    #"./datafiles/ontology_poetry.ttl"
    './datafiles/dbpedia_2016-10.owl'
    ]

    model = "sentence-transformers/all-mpnet-base-v2"

    ontology_db = GraphRag(model, ontology_files)

    print(ontology_db.process_query('give me all the authors that were born in the second century', 5))
    print('done')
    