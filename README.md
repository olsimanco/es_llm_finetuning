# Evolutionary Strategies (ES) for LLM Fine-Tuning

## Overview
This repository contains a robust framework for the practical implementation of artificial intelligence, focusing specifically on fine-tuning Large Language Models (LLMs) using Evolutionary Strategies (ES). It provides a comprehensive suite of tools for model training, prompt decoding, and accuracy validation, utilizing various ES algorithms such as BPB and OLMES[cite: 1]. 

Designed with modularity in mind, this project allows researchers and data engineers to experiment with non-gradient-based optimization methods for LLMs.

## Repository Structure

The project is organized into several key directories and modules:

* **Core Scripts:**
  * `download_model.py`: Handles the acquisition and local caching of the base language models[cite: 1].
  * `make_colab.py`: Utility script to quickly adapt the environment for execution within Google Colab[cite: 1].
  * `es_finetune_project.ipynb`: An interactive Jupyter Notebook demonstrating the end-to-end pipeline[cite: 1].

* **`src/` - Source Code:**[cite: 1]
  * `es_trainer.py`, `es_trainer_bpb.py`, `es_trainer_olmes.py`: The primary training loops implementing standard ES, BPB, and OLMES methodologies[cite: 1].
  * `olmes_wrapper.py`, `full_olmes_wrapper.py`, `bpb_wrapper.py`: Algorithm-specific wrappers designed to manage evolutionary state and model interactions[cite: 1].
  * `baseline_sft.py`: Provides a baseline Supervised Fine-Tuning implementation for performance comparison[cite: 1].
  * `accuracy_validator.py`: Evaluates model performance and ensures output quality against established benchmarks[cite: 1].
  * `generate_from_prompt.py` & `decode_prompt.py`: Utilities for handling prompt inputs and decoding model generations[cite: 1].
  * `config.py`: Centralized configuration management for hyperparameters and environment variables[cite: 1].

* **`results/` - Experiment Tracking:**[cite: 1]
  * Contains detailed logging and generation statistics across various runs (e.g., `Run_01_DetailedStats_20260216_1510`, `es_res_real_olmes`)[cite: 1].
  * Stores outputs such as `best_prompt_overall.txt`, `log.csv`, and `config.json` for reproducible research[cite: 1].
  * `leaderboard.csv`: Tracks and compares the performance metrics of different models and algorithmic approaches[cite: 1].

## Setup & Installation

For development, a Linux environment (such as Ubuntu) is highly recommended. Visual Studio Code serves as an excellent primary text editor for navigating this project's architecture.

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd es_llm_finetuning-main
