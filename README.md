# FeelWise: Emotion Classification Model

FeelWise is a transformer-based emotion classification model built with PyTorch and integrated with Hugging Face Transformers. This model classifies text into 6 different emotion categories using a custom encoder architecture.

## 🌟 Features

- **Custom Transformer Architecture**: Built from scratch with multi-head attention mechanisms
- **Emotion Classification**: Classifies text into 6 emotion categories
- **Hugging Face Integration**: Fully compatible with Hugging Face's AutoModel and AutoTokenizer
- **Pre-trained Model**: Available on Hugging Face Hub at `sanjithrj/FeelWise`
- **Easy Inference**: Simple API for making predictions on text

## 📋 Model Architecture

The FeelWise model is based on a custom transformer encoder architecture with the following components:

- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: Configurable number of attention heads (default: 8)
- **Feed-Forward Networks**: Two-layer FFN with ReLU activation
- **Layer Normalization**: Applied after each sub-layer
- **Classification Head**: Final linear layer for emotion classification

### Default Configuration

```python
d_model = 256          # Model dimension
max_len = 500          # Maximum sequence length
input_vocab_size = 50000  # Vocabulary size
n_layers = 1           # Number of encoder layers
n_head = 8             # Number of attention heads
d_ff = 1024            # Feed-forward network dimension
num_classes = 6        # Number of emotion classes
dropout = 0.1          # Dropout rate
```

## 🚀 Installation

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers library

### Install Dependencies

```bash
pip install torch transformers huggingface_hub
```

## 💻 Usage

### Loading the Model

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load the tokenizer and model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("sanjithrj/FeelWise")
model = AutoModel.from_pretrained("sanjithrj/FeelWise", trust_remote_code=True)
```

### Making Predictions

```python
import torch.nn.functional as F

# Input text
input_text = "I am feeling happy today!"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=50)

# Get model predictions
with torch.no_grad():
    outputs = model(inputs["input_ids"])
    probs = F.softmax(outputs, dim=-1)
    predicted_emotion = torch.argmax(probs, dim=-1)

print(f"Predicted emotion class: {predicted_emotion.item()}")
print(f"Probabilities: {probs}")
```

### Using the Model Locally

If you want to use the model from local files:

```python
from FWModule.config import FeelWiseConfig
from FWModule.model import FeelWiseModel
import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/Tokenizer")

# Initialize model with configuration
config = FeelWiseConfig()
model = FeelWiseModel(config)

# Load trained weights
weights = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(weights['state_dict'])
model.eval()
```

## 📁 Project Structure

```
HF/
├── FWModule/              # Model implementation
│   ├── __init__.py
│   ├── config.py         # Model configuration
│   └── model.py          # Model architecture
├── Tokenizer/            # Tokenizer files
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── main.py               # Script to push model to Hugging Face Hub
├── inference.py          # Inference example script
├── model.pth            # Trained model weights
└── README.md            # This file
```

## 🔧 Pushing to Hugging Face Hub

To push your own version of the model to Hugging Face Hub:

```python
from huggingface_hub import login
from FWModule.config import FeelWiseConfig
from FWModule.model import FeelWiseModel
from transformers import AutoTokenizer

# Login to Hugging Face
login('your_huggingface_token')

# Load model and tokenizer
config = FeelWiseConfig()
model = FeelWiseModel(config)
tokenizer = AutoTokenizer.from_pretrained("path/to/Tokenizer")

# Load weights
weights = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(weights['state_dict'])

# Register for auto classes
config.register_for_auto_class()
model.register_for_auto_class("AutoModel")

# Push to hub
config.push_to_hub('YourModelName')
model.push_to_hub('YourModelName')
tokenizer.push_to_hub('YourModelName')
```

## 🎯 Emotion Classes

The model classifies text into 6 emotion categories. The exact mapping of class indices to emotion labels depends on your training data.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- **Hugging Face Model**: [sanjithrj/FeelWise](https://huggingface.co/sanjithrj/FeelWise)
- **GitHub Repository**: [sanjithwoxsen/HF](https://github.com/sanjithwoxsen/HF)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Make sure to update your Hugging Face token in `main.py` if you plan to push models to the hub.
