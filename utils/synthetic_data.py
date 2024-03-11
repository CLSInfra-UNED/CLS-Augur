import os
import pandas as pd

from rdflib import Graph
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
EMB_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 2048
TEMP = 0

dbpedia_graph = Graph().parse('../datafiles/dbpedia_2016-10.owl', format='xml')

dbpedia_ids = {element for triple in dbpedia_graph for element in triple}