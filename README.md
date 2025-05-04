# AI Doctor 3: Medical Expert Finetuning

This project demonstrates how to finetune the DeepSeek-R1-Distill-Llama-8B model to create an AI medical assistant capable of answering complex medical questions with clinical reasoning.

## Overview

The AI Doctor 3 project leverages the Unsloth library to efficiently finetune a large language model (LLM) on medical datasets. The model is specifically trained to:

1. Apply clinical reasoning to medical questions
2. Generate detailed step-by-step thought processes
3. Provide accurate medical diagnoses and treatment recommendations
4. Follow a chain-of-thought reasoning approach

## Features

- Finetuning of DeepSeek-R1-Distill-Llama-8B (8B parameter model)
- Integration with Weights & Biases for experiment tracking
- Implementation of LoRA (Low-Rank Adaptation) for efficient training
- Chain-of-thought reasoning with explicit thinking steps
- Training on the FreedomIntelligence/medical-o1-reasoning-SFT dataset

## Setup & Requirements

### Dependencies

- Python 3.8+
- PyTorch
- Transformers
- Unsloth
- trl (Transformer Reinforcement Learning)
- Weights & Biases (wandb)
- Hugging Face libraries

### Environment Setup

1. Install the required packages:
   ```
   pip install unsloth
   pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
   ```

2. Set up API tokens:
   - Create a Hugging Face token for model access
   - Create a Weights & Biases token for experiment tracking

## Usage

### Running in Jupyter Notebook

The project includes a Jupyter notebook (`AI_Doctor_3.ipynb`) with all the steps:

1. Setup and import libraries
2. Load the pre-trained DeepSeek-R1 model
3. Test the model with inference before finetuning
4. Load and preprocess the medical dataset
5. Setup LoRA for efficient finetuning
6. Train the model
7. Evaluate the fine-tuned model

### Python Script

Alternatively, you can use the provided Python script (`ai_doctor_3.py`) which contains the same functionality as the notebook.

## Model Training Details

- **Base Model**: DeepSeek-R1-Distill-Llama-8B
- **Training Epochs**: 1
- **Learning Rate**: 2e-4
- **Batch Size**: 2 (with gradient accumulation steps of 4)
- **Sequence Length**: 2048 tokens
- **LoRA Config**: r=16, alpha=16, targeting attention and MLP layers

## Example Usage

```python
# Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path_to_saved_model",
    max_seq_length=2048,
    load_in_4bit=True
)

# Define a test question
question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or
              sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings,
              what would cystometry most likely reveal about her residual volume and detrusor contractions?"""

# Set up prompt
prompt = """
Below is a task description along with additional context provided in the input section. Your goal is to provide a well-reasoned response that effectively addresses the request.

Before crafting your answer, take a moment to carefully analyze the question. Develop a clear, step-by-step thought process to ensure your response is both logical and accurate.

### Task:
You are a medical expert specializing in clinical reasoning, diagnostics, and treatment planning. Answer the medical question below using your advanced knowledge.

### Query:
{}

### Answer:
<think>{}
"""

# Inference
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Answer:")[1])
```

## License

This project is for educational and research purposes. Please ensure responsible use of AI medical assistants, as they should supplement, not replace, professional medical advice.

## Acknowledgements

- [DeepSeek AI](https://github.com/deepseek-ai) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for optimization libraries
- [FreedomIntelligence](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) for the medical dataset 