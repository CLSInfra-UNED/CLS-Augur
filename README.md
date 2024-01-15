# CLS-Augur
Natural Language SparQL Query Translator

## Overview
This project integrates ontology management and conversational AI, using Retrieval-Augmented Generation (RAG) and advanced model optimizations. It is designed for translating natural language queries into structured SPARQL queries, leveraging a conversational AI framework.

## Key Components
- **RAG for RDF Graphs**: Uses `GraphRag` and `SparQLRag` classes to manage RDF graphs and vector databases.
- **Conversational AI Pipeline**: Implements a pipeline using quantized models for efficient conversational AI.
- **Dynamic Prompt Generation**: Generates prompts dynamically for conversational agents to facilitate SparQL query generation.
- **Main Application**: Integrates all components to process conversational queries and generate SPARQL queries.
- **Evaluation**: Implements evaluation functions for execution accuracy (EX) and exact-set-match accuracy (EM)

## Installation
Ensure that you have Python 3.10+ installed. Then, clone this repository and install the required dependencies:
```bash
TODO
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


