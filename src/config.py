# Configuration for the Evolutionary Strategy
class ESConfig:
    # Model Settings
    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda"

    # Evolution Hyperparameters
    population_size = 20  # How many "children" to spawn per step
    sigma = 0.02  # How much "noise" to add (mutation strength)
    learning_rate = 0.01  # How fast the model learns
    generations = 100  # How many loops to run

    # Task Settings
    task_name = "minerva_math_algebra:bpb::olmes"
