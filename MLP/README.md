# Multi-Layer Perceptron (MLP) Character Language Model

An advanced character-level language model utilizing a Multi-Layer Perceptron (MLP) architecture built from scratch in PyTorch. This model implements a fixed-window context approach (inspired by Bengio et al. 2003) to predict the next character in a sequence.

> **Project Philosophy:** This repository marks the second major milestone in my deep learning journey. It is strictly an **educational project** focused on understanding the core mechanics of feature embeddings, layer dimensions, and model evaluation. Instead of using black-box abstractions, I built the network layers and data pipelines from first principles to master tensor transformations and gradient propagation.

---

## Learning Objectives & Milestones

### 1. Advancing into Neural Network Architectures
Moving past the simple 2-character transition limits of the Bigram model, this project introduces much more powerful architectural concepts:
* **Character Embeddings:** Learned how to map discrete characters into a continuous, low-dimensional vector space using an embedding lookup table matrix.
* **Context Windows & Flattening:** Implemented a rolling context window (e.g., using the previous 3 characters to predict the 4th) and mastered the tensor manipulation required to flatten multi-dimensional embeddings (`.view(-1, ...)`).
* **Hidden Layers & Activations:** Implemented non-linear hidden layers utilizing the **Hyperbolic Tangent (`tanh`)** activation function to allow the network to learn complex, non-linear character patterns.
* **Numerical Stability:** Explored the shift from manual exponentiation to PyTorch's `F.cross_entropy`, understanding why it is mathematically superior due to its protection against numerical overflow and underflow.

### 2. Rigorous Evaluation & Engineering Workflows
* **Train / Validation / Test Splits:** Implemented proper dataset partitioning (80% Train, 10% Val, 10% Test) to scientifically monitor the model for **underfitting** or **overfitting**.
* **Hyperparameter Tuning:** Explored learning rate schedules, batch sizing (Mini-batch Gradient Descent), and scale initialization for weights and biases.
* **Git Version Control:** Scaled my central learning repository by adding this independent folder structure, managing modular code files, and tracking updates cleanly through VS Code.

---

## Evaluation & Comparative Observations

* **Bigram Baseline Loss:** ~2.50 (Limited by only looking at 1 character of context).
* **MLP Training Loss:** Successfully brought the loss down significantly lower than the bigram model by taking advantage of multi-character context windows and continuous multi-dimensional embeddings.
* **Overfitting Insights:** By comparing the *Training Loss* against the *Validation Loss*, I learned how to identify when the network starts to memorize the dataset rather than generalizing linguistic patterns.