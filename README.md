# Gestr-JEPA: Indian Sign Language Recognition with Le-JEPA

This model is a lightweight **Joint Embedding Predictive Architecture (JEPA)** designed to recognize Indian Sign Language (ISL) characters from a sensory glove. 

It was trained using **Le-JEPA** (Latent-Space JEPA) with **SIGReg** (Sketched Isotropic Gaussian Regularization), allowing it to learn a highly structured, spherical latent space directly from noisy sensor readings without needing negative pairs or contrastive learning.

### Model Details
- **Architecture:** MLP-based Encoder-Predictor
- **Input:** 16-dimensional vector (5 Flex Sensors + 3 IMU axes $\times$ 2 hands)
- **Latent Dimension:** 64
- **Hidden Dimension:** 256
- **Regularization:** SIGReg (Sketched Isotropic Gaussian Regularization)
- **Downstream Task:** 26-way Classification (A-Z) via Linear Probe

### How to Use

You can load this model directly using the Hugging Face `transformers` library. Since this uses a custom architecture, you must enable `trust_remote_code=True`.

#### 1. Install Dependencies
```bash
pip install torch transformers
```

#### 2. Inference Code

This script loads the model, passes in a random sensor reading, and prints the internal representations and final prediction.

```python
import torch
from transformers import AutoModelForSequenceClassification

# 1. Load the model from the Hub
# 'trust_remote_code=True' is required for custom architectures like Gestr-JEPA
model_id = "sattwik21/gestr-jepa-isl"
print(f"Loading model from {model_id}...")

model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    trust_remote_code=True
)
model.eval()

# 2. Create Dummy Input (Simulating the Glove)
# Shape: (Batch_Size=1, Num_Sensors=16)
sensor_readings = torch.randn(1, 16)

# 3. Run Inference
with torch.no_grad():
    # The model forward pass returns a dict containing loss, logits, and hidden_states
    outputs = model(sensor_values=sensor_readings)
    
    # Extract components
    encodings = outputs["hidden_states"]  # The JEPA latent representation (z)
    logits = outputs["logits"]            # The Linear Probe output
    
    # Get Prediction
    predicted_idx = torch.argmax(logits, dim=1).item()
    predicted_letter = model.config.id2label[predicted_idx]

# 4. Print Results
print("-" * 30)
print(f"Input Shape:     {sensor_readings.shape}")
print(f"Encoding Shape:  {encodings.shape}  <-- 64-dim spherical latent vector")
print(f"Logits Shape:    {logits.shape}     <-- Scores for 26 classes")
print("-" * 30)
print(f"Predicted Letter: {predicted_letter}")
```
