# Rapid Cross-Domain Transfer in Wearable Systems via Unsupervised Latent Space Prediction

This repository contains the official implementation of the hardware-agnostic kinematic encoder based on the **Joint Embedding Predictive Architecture (JEPA)** presented in our IJCAI 2026 submission. **SenseJEPA** (project name: `sensejepa`) is designed for wearable gesture recognition, enabling rapid adaptation to new sign languages and users with minimal labeled data.

## 🌟 Key Features

* **Predictive Latent Modeling**: Unlike contrastive or generative self-supervised learning, our JEPA-based approach predicts latent representations, prioritizing high-level semantic movement over raw sensor noise.


* **SIGReg Regularization**: Utilizes Sketched Isotropic Gaussian Regularization (SIGReg) to prevent representation collapse and enforce an information-maximizing distribution in the latent space, replacing unstable heuristic-based weight updates.


* **High Data Efficiency**: Achieves a **17.22% accuracy improvement** over fully supervised baselines on American Sign Language (ASL) recognition when using only **1% of the training data**.


* **Hardware Agnostic**: Includes a robust tokenization and masking strategy for sparse, heterogeneous sensor streams (Flex sensors and IMUs).



## 🏗️ Architecture

The system processes data through a structured pipeline:

1. **Filtering**: Raw signals pass through Low-Pass (LPF), Oversampling, and Drift-removal filters.


2. **Tokenization**: A linear tokenizer converts sensor readings into -dimensional tokens with learnable positional embeddings.


3. **Sensor Masking**: A random masking technique drops a ratio of tokens to ensure the learned latents are robust to sensor malfunctions.


4. **JEPA Pre-training**: The encoder is pre-trained on biomechanically complex primitives like **Kathakali Mudras** and **Ninja Kuji-in** hand signs to capture the universal hand kinematics manifold.



## 🛠️ Installation

This project uses `uv` for dependency management. Ensure you have Python 3.12+ installed.

```bash
# Clone the repository
git clone https://github.com/sattwik-sahu/sensejepa.git
cd sensejepa

# Install dependencies using uv
uv sync

```

### Core Dependencies

* `torch >= 2.9.1`
* `transformers >= 4.57.3`
* `x-transformers >= 2.12.2`
* `timm >= 1.0.24`
* `scikit-learn >= 1.8.0`




## 💻 Usage (Inference Example)

You can use the encoder as a feature extractor for downstream tasks. Below is a conceptual example of loading the components defined in `src/sensejepa`:

```python
import torch
from sensejepa.utils.modules.tokenizer import SensorTokenizer
from sensejepa.utils.modules.encoder import SensorEncoder # Assuming standard transformer encoder

# Initialize components (d=128 as per paper)
tokenizer = SensorTokenizer(n_sensors=16, dim=128)
encoder = SensorEncoder(dim=128, n_heads=8, n_layers=4)

# Simulated input: (Batch_Size=1, Num_Sensors=16)
# Representing 5 Flex + 3 IMU axes per hand
sensor_readings = torch.randn(1, 16)

# Tokenize and Encode
tokens = tokenizer(sensor_readings)
latents = encoder(tokens) # Extracted kinematic features

print(f"Latent representation shape: {latents.shape}")

```



## 📊 Experimental Results

Pre-training on complex gestures significantly improves transferability to American Sign Language (ASL) recognition, especially in low-data regimes.

| Training Data Fraction | Fully Supervised | **JEPA (Pre-trained Frozen)** | **Improvement (Δ)** |
| --- | --- | --- | --- |
| **1.0%** | 25.13% | **42.35%** | **+17.22%** |
| **10.0%** | 35.02% | **44.95%** | **+9.93%** |
| **25.0%** | 49.93% | **66.60%** | **+16.67%** |
| **100.0%** | 92.92% | 87.30% | -5.62% |

---

> [!NOTE]
> The data and training code will be released soon.
