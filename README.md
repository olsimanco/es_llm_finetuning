# Evolutionary Strategies (ES) for LLM Fine-Tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Model: Qwen2.5-0.5B](https://img.shields.io/badge/Model-Qwen2.5--0.5B-orange)](https://huggingface.co/Qwen/Qwen2.5-0.5B)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository explores the use of Evolutionary Strategies (ES) as a gradient-free alternative to backpropagation-based fine-tuning for Large Language Models (LLMs)[cite: 2]. By optimizing parameter spaces directly through perturbed parameter vectors and fitness evaluations, this approach bypasses the need for differentiable loss surfaces[cite: 2]. The experiments focus on Qwen2.5-0.5B, evaluating performance on mathematical reasoning (GSM8K) and common-sense completion (HellaSwag) datasets[cite: 2]. 

## Methodology
The project implements several optimization structures to evaluate the efficacy of ES in continuous parameter spaces.

### 2.1 Soft-Prompt Fine-Tuning
This approach freezes the base model parameters and optimizes a continuous embedding vector (the soft prompt) prepended to the input[cite: 2]. 
*   **Prompt Integration:** The soft prompt matrix $P \in \mathbb{R}^{T \times d}$ is concatenated with the input embeddings $E_{input} \in \mathbb{R}^{L \times d}$ to form $E_{combined} = [P || E_{input}] \in \mathbb{R}^{(T+L) \times d}$[cite: 2].
*   **Optimization:** The matrix is flattened into a 1D vector and optimized using the OpenAI-ES framework[cite: 2].
*   **Fitness Function (GSM8K):** Uses Bits-Per-Byte (BPB) negative cross-entropy loss[cite: 2]. The prompt tokens are masked so the signal strictly isolates the model's predictive accuracy on target tokens[cite: 2]. The fitness is defined as:
    $$F_{BPB}(\theta) = -\frac{1}{M}\sum_{j=1}^{M}\mathcal{L}_{CE}(P(\theta) \oplus x_j, a_j)$$[cite: 2]

### 2.2 Layer-Wise Fine-Tuning
Alternatively, ES is applied to a targeted subset of internal model weights while the rest of the model remains frozen[cite: 2]. This includes experimenting with specific decoder layers, isolating attention blocks, or targeting MLP layers to reduce the dimensionality of the optimization problem[cite: 2].

## Results and Benchmarks
The soft-prompting approach yielded the following empirical metrics across 100 generations[cite: 2]:

*   **GSM8K (Mathematical Reasoning):**
    *   Best Accuracy: 7.6%[cite: 2]
    *   Best BPB Score (Loss): -0.6742[cite: 2]
*   **HellaSwag (Common Sense Reasoning):**
    *   Best Accuracy: 28.0%[cite: 2]
    *   Best Margin Score: -0.3729[cite: 2]

## Architecture and Repository Structure
*   `src/es_trainer_bpb.py` & `src/es_trainer_olmes.py`: Core evolutionary algorithms orchestrating population spawning and weight updates[cite: 1].
*   `src/bpb_wrapper.py`: Handles batching, masking, and loss calculations for the ES loop[cite: 1].
*   `src/accuracy_validator.py`: Executes forward passes and parses string outputs to verify exact-match correctness[cite: 1].
*   `src/baseline_sft.py`: A Supervised Fine-Tuning (LoRA) baseline used for performance and efficiency comparisons[cite: 1].
*   `results/`: Automated experiment tracking directory, logging generation statistics and saving the best evolved prompt tensors (`.pt`)[cite: 1].

## Setup and Execution

**1. Environment Setup**
Ensure you have Python 3.8+ installed. A Linux environment is highly recommended.
```bash
git clone <your-repository-url>
cd es_llm_finetuning-main
pip install -r requirements.txt# Evolutionary Strategies (ES) for LLM Fine-Tuning

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
