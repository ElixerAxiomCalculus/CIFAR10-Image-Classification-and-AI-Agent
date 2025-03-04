# CIFAR-10 Image Classification & AI Agent with RAG

This repository contains implementations of two tasks:
1. **CIFAR-10 Image Classification using Deep Learning**
2. **AI Agent with RAG and Function Calling**

## Task 1: CIFAR-10 Image Classification

### Objective
Train a deep learning neural network to classify images using the CIFAR-10 dataset, which contains 60,000 labeled images of 10 different classes.

###  Dataset
- CIFAR-10 Dataset: Imported from **TensorFlow/Keras datasets**.

###  Features Implemented
- **Data Preprocessing**:
  - Loaded the dataset directly from TensorFlow/Keras.
  - Normalized pixel values to the range [0,1].
  - Split the dataset into training and test sets.
- **Built a CNN Model using:**
  - Conv2D, MaxPooling2D, Flatten, Dense, Dropout, and Batch Normalization layers.
- **Training & Validation**:
  - Applied proper train-test split.
  - Used early stopping to prevent overfitting.
  - Tuned hyperparameters such as learning rate and batch size for better performance.
- **Performance Evaluation**:
  - Computed Accuracy, Precision, Recall, and F1-score.
  - Plotted training and validation accuracy/loss curves to analyze model performance.

## Task 2: AI Agent with RAG and Function Calling
Important : I couldn't run the Hugging Face API because it is taking time to take my submission approval for Meta LLama 2 API integration access
###  Objective
Develop an AI agent capable of processing files as input using **LlamaIndex** or **LangChain**. The agent should exhibit reasoning capabilities, support **Retrieval-Augmented Generation (RAG)**, and implement **function calling** for arithmetic operations.

###  Features Implemented
- **File Ingestion** using **LlamaIndex** or **LangChain**.
- **Retrieval-Augmented Generation (RAG)** to enhance responses with retrieved information.
- **Function Calling**:
  - Arithmetic operations: Addition, Subtraction, Multiplication, Division.
- **Reasoning Capabilities**:
  - The agent processes and infers information from provided files.

## Usage
### CIFAR-10 Classification
Run the notebook `GDSC_Task_1.ipynb` to train and evaluate the CNN model.

### AI Agent with RAG
Run the notebook `GDSC_Task_2.ipynb` to interact with the AI agent.

