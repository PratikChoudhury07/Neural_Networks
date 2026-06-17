# Character-Level Bigram Language Model from Scratch

A foundational deep learning project built to explore the raw mechanics of neural networks and master the essentials of Git/GitHub version control. 

> **Project Philosophy:** This repository is explicitly created for **educational and learning purposes**, not as a production-ready product. Instead of relying on high-level abstraction frameworks (like PyTorch's built-in `nn.Linear` or `nn.CrossEntropyLoss`), everything here is built manually from first principles. The goal was to peel back the layers of a modern language model to understand the raw math, while simultaneously learning how to manage code repositories professionally.

---

## Learning Objectives & Milestones

### 1. Diving Into Neural Networks
Instead of treating deep learning like a "black box," this project focuses on understanding exactly how data flows through a network:
* **The Softmax Function From Scratch:** Implemented the mathematical equivalent of a Softmax layer using raw PyTorch tensors (`.exp()` and row-wise `.sum(1, keepdims=True)` normalization via tensor broadcasting).
* **Loss Function Mechanics:** Deepened my understanding of **Negative Log-Likelihood (NLL)** loss, learning how probabilities are mapped to a log scale to prevent numerical underflow in computer memory.
* **Manual Optimization Loops:** Wrote a complete training loop from scratch, manually executing the classic pipeline: Forward Pass ➡️ Loss Calculation ➡️ Backward Pass (`loss.backward()`) ➡️ Stochastic Gradient Descent (SGD) Parameter Updates.
* **Regularization (L2):** Implemented Weight Decay (`0.01 * (w**2).mean()`) to observe how penalizing large weights acts as a data smoother, preventing the model from becoming overconfident.

### 2. Mastering Git & GitHub Workflows
Beyond the machine learning math, this project served as a playground for learning standard software engineering tools:
* **Repository Management:** Learning how to structure files, manage datasets, and write clean documentation for the open-source community.
* **VS Code Integration:** Utilizing integrated Source Control tools to initialize repositories, stage changes, write clean commit messages, and publish branches directly to GitHub.

---

## Key Technical Concepts Implemented
* **Data Processing:** Constructed character vocabulary mappings (`stoi` & `itos`) dynamically. Optimized memory handling by avoiding massive intermediate string concatenations during vocabulary extraction.
* **Neural Network Architecture:** Designed a single-layer neural network using **One-Hot Encoding** and **Matrix Multiplication (`matmul`)** to predict the next character's logits.
* **Text Generation:** Developed an autoregressive text sampler utilizing `torch.multinomial` to probabilistically sample the next character based on a predictable random seed generator.

## Evaluation
* **Initial Random Loss:** ~3.64 (comparable to a completely uniform $1/27$ random distribution).
* **Trained Loss:** Successfully minimized the loss over a manual training regime, proving that even a single-layer network can successfully extract statistical linguistic patterns from a raw text file.