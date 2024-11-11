# AgriBench: A Benchmark for Large Language Models (LLMs) in Agriculture

AgriBench is a project that aims to establish a benchmark specifically tailored for evaluating the performance of Large Language Models (LLMs) in the agricultural domain. This benchmark focuses on various natural language processing (NLP) tasks relevant to agriculture, utilizing structured data, ontology-based methods, and automatic question-answer pair generation.

## Key Features

- **Structured Datasets**: Integration of agricultural knowledge bases and datasets such as GBIF, EPPO, and AGROVOC. The benchmark leverages relational databases and ontologies for NLP tasks.
  
- **Automatic Q&A Generation**: Converts database queries and results into natural language. Incorporates a quality metric to filter outputs, enabling automatic question-answer generation for complex agricultural queries.

- **Task Diversity**: Covered tasks include:
  - Text classification
  - Sentiment analysis
  - Question answering (Q&A)
  - Text generation

- **Ontology-based Methods**: Utilizes agricultural ontologies like **AGROVOC (FAO)**, allowing models to reason over structured, domain-specific knowledge.

- **Retrieval Augmented Generation (RAG)**: Incorporates relational databases into LLMs for improved information retrieval and response generation.

## Goals

- Provide a **standardized evaluation protocol** for assessing LLMs' agricultural knowledge and performance.
- Offer a **versatile dataset** that reflects real-world agricultural applications.
- Enhance the accuracy and performance of LLMs in tasks related to **agribusiness, sustainability, and agroecology**.

## How it Works

1. **Data Preparation**: Scraping and formatting structured datasets and converting them into accessible formats for LLM processing.
2. **Model Integration**: Incorporates advanced features like Relational Database integration and Ontology-based reasoning to improve context understanding.
3. **Evaluation and Filtering**: Apply a quality metric system to filter generated data and evaluate the model's performance.

## Future Directions

- **Expansion of Agricultural Datasets**: Continuously adding more structured datasets for broader coverage.
- **Improved NLP Techniques**: Exploration of additional AI methods to further refine agriculture-specific natural language understanding.
