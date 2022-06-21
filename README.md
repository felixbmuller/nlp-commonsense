# Commonsense Reasoning in Natural Language Processing

Assignments for UBC's NLP Commonsense course (CPSC 532V), taught by Vered Shwartz.

## Assignment 1

Mainly implementing a search function that finds relevant paths in the ConceptNet knowledge between two given terms. Includes term extraction, normalization etc.

See [notebooks/Assignment1.ipynb](notebooks/Assignment1.ipynb) and [src/prepare_data.py](src/prepare_data.py).

## Assignment 2

The goal was to add commonsense knowledge extracted via the path search from assignment 1 to aid a question answering model. The model is a finetuned BERT model that receives the most relevant ConceptNet-paths as part of its input. I trained and evaluated both a baseline and a knowledge base model model and performed a qualitative and quantitative error analysis on the COPA dataset.

See [notebooks/NLP_Commonsense_Assignment_2_KB_Model.ipynb](notebooks/NLP_Commonsense_Assignment_2_KB_Model.ipynb) for my model and [notebooks/NLP_Commonsense_Assignment_2_Baseline_Model.ipynb](notebooks/NLP_Commonsense_Assignment_2_Baseline_Model.ipynb) for the baseline.
