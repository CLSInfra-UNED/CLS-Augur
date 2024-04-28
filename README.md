# CLS-Augur: Natural Language to SparQL Query Translator

## Overview
This project integrates ontology management and conversational AI, using Retrieval-Augmented Generation (RAG) and advanced model optimizations. It is designed for translating natural language queries into structured SPARQL queries, leveraging a conversational AI framework.

## Key Components
- **RAG for RDF Graphs**: Uses `GraphRag` and `SparQLRag` classes to manage RDF graphs and vector databases.
- **Conversational AI Pipeline**: Implements a pipeline using quantized models for efficient conversational AI.
- **Dynamic Prompt Generation**: Generates prompts dynamically for conversational agents to facilitate SparQL query generation.
- **Main Application**: Integrates all components to process conversational queries and generate SPARQL queries.
- **Evaluation**: Implements evaluation functions for execution accuracy (EX) and exact-set-match accuracy (EM)

## Installation
Ensure that you have Python 3.10+ installed. Then, clone this repository and install the required dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
To run the main application:
```bash
python main.py
```

## Main Functions

### `GraphRag` and `SparQLRag` classes
- Manages RDF graphs and vector databases.
- `GraphRag`: Loads RDF files and processes them into a vector database.
- `SparQLRag`: Work-in-progress for implementing RAG with SPARQL queries.

### Agents
We use two agents to extract and define relevant entities and relations from natural language queries, which is essential for leveraging our enriched ontology schema. These agents employ different approaches: PoS-based extraction and LLM-based extraction. They play a crucial role in converting user queries into structured data that aligns with the RDF environment, making the detailed ontology schema more accessible and usable.

#### PoS-based extraction
For this technique, we employ Stanford's Stanza model to extract relevant entities and relations from natural language queries. This model is a robust NLP toolkit that facilitates the extraction process by accurately identifying entities and their relations within the input text. Through syntactic and semantic analysis, Stanza identifies nouns, verbs, and interrogative pronouns and adverbs PoS tags, thereby capturing the structural and semantic elements of the query. Once the relevant data is extracted, it is passed to a LLM for further processing. Specifically, the LLM generates concise definitions for each entity and relation extracted from the query. This process is aimed at retrieving relevant information from the RAG, as previously explained.

#### LLM-based extraction
In the second technique, we solely rely on a LLM to extract and create definitions for all relevant entities and relations present in the natural language queries. By leveraging the contextual understanding and knowledge representation capabilities of LLM, this approach ensures that the extracted data is transformed into a structured format suitable for subsequent query processing within the RDF environment, bypassing the need for a separate tool such as Stanza for initial data extraction. This streamlined process enhances efficiency and coherence in the generation of definitions, facilitating seamless retrieval of relevant information from the RAG. The LLM directly processes the input queries, discerning the entities and relations based on the contextual information provided. Subsequently, it generates succinct definitions for each entity and relation identified in the query. This end-to-end approach not only streamlines the extraction process but also ensures consistency and coherence in the definitions generated, as they are crafted within the same model that comprehends the query. This method offers a seamless and efficient solution for extracting and defining entities and relations from natural language queries in the context of NLIB systems. Again, this process is conducted with the objective of retrieving relevant information from the RAG, as previously elucidated.

### Conversational Pipeline (`model.conversational_pipeline`)
- Sets up a pipeline for conversational AI using a quantized language model.

### Dynamic Prompt Generation (`ctx.prompt_llama_code`)
- Generates conversational prompts tailored to the user query and context, aggregating Chain of Though, 
in-context learning examples and Retrieval Augmented Generation

## Example
The script in `main.py` demonstrates the application's functionality:
- Initializes the ontology and queries databases.
- Creates a conversational pipeline.
- Processes a test suite of queries through this pipeline.

## Results

Accuracy and Validity Evaluation Across Models and Methods:
| Model/Method         | F1    | Recall | Precision | Jaccard | Acc  | Valid |
|----------------------|-------|--------|-----------|---------|------|-------|
| **DeepSeek-6.7B/Naive** | 12.98 | 66.48  | 7.27      | 7.12    | 29.9 | 80.7  |
| **DeepSeek-6.7B/PoS**   | 13.65 | <u>67.16</u> | 7.67      | 7.51    | 34.0 | 97.5  |
| **DeepSeek-6.7B/Gen**   | 14.79 | 66.95  | 8.41      | 8.22    | 35.2 | 97.5  |
| **DeepSeek-6.7B/FSL**   | **20.87** | 65.05  | **12.70**    | **12.25**  | <u>37.4</u> | <u>98.4</u> |
| **CodeLlama-13B/Naive** | 12.98 | 66.48  | 7.27      | 7.12    | 33.2 | 95.8  |
| **CodeLlama-13B/PoS**   | 13.71 | 66.39  | 7.72      | 7.55    | 37.8 | 98.8  |
| **CodeLlama-13B/Gen**   | 14.18 | <u>67.03</u> | 8.02      | 7.85    | 37.8 | 98.7  |
| **CodeLlama-13B/FSL**   | **20.87** | 65.05  | **12.70**    | **12.25**  | <u>41.0</u> | <u>99.5</u> |
| **ChatGPT 3.5/Naive**   | 12.98 | 66.48  | 7.27      | 7.12    | 34.3 | 75.5  |
| **ChatGPT 3.5/PoS**     | 13.76 | 67.03  | 7.75      | 7.58    | 39.4 | 99.0  |
| **ChatGPT 3.5/Gen**     | 14.45 | **67.57** | 8.19      | 8.01    | 38.6 | **99.6** |
| **ChatGPT 3.5/FSL**     | **20.87** | 65.05  | **12.70**    | **12.25**  | **47.7** | 99.5  |

- Bold numbers indicate the best performance in each column.
- Underlined numbers indicate the second-best or notable performances.