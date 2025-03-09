# LLM-Text-Classification
Apply MiniLM to do 4-dimension Text Classification
# **Quantum Computing Text Classification with MiniLM**

## **Project Overview**
This project classifies quantum computing research papers into four categories using a **fine-tuned MiniLM transformer model**. Traditional TF-IDF + ML models were insufficient for handling the complexity of scientific terminology, making **LLMs a better choice**.

### **Categories**
1. **Models of Manipulating Qubits for Computation**
2. **Methods of Building Qubits**
3. **Addressing Obstacles to Quantum Computation**
4. **Applications of Quantum Computing**

---

## **Files & Directory Structure**

| File Name                  | Description |
|----------------------------|-------------|
| **Text_Classification.py**  | Full script for training and predicting categories |
| **test_script.py**         | Debugging script for model evaluation |
| **test_report.pdf**        | Report detailing methodology and results |
| **testing_results.csv**    | Test set with predicted categories |
| **Resume_Zhihao_Yang.pdf** | Author's resume |

---

## **Methodology**

### **1. Model Selection**
Given a **small dataset (~500 rows)** and **NVIDIA RTX 4060 GPU**, the following models were compared:

| Model                | Parameters | Small Data Suitability | Computational Efficiency |
|----------------------|------------|------------------------|-------------------------|
| **MiniLM** (Selected) | 22M        | ✅ Excellent | ✅ Fast inference |
| DistilBERT          | 66M        | ⚠️ Needs more data | ⚠️ Slower |
| TinyBERT            | 14M        | ✅ Very fast | ⚠️ Lower accuracy |
| DeBERTa-v3-base     | 86M        | ❌ Needs >10K samples | ❌ GPU-intensive |

MiniLM was chosen for **efficiency and performance**.

### **2. Data Preprocessing**
- **Stopword Removal**: Removed "of," "for," "to," "in," "the," "and".
- **Character Standardization**: Removed special characters and spaces.

### **3. Data Augmentation**
- **Synonym Replacement** (only for **nouns**) to maintain scientific accuracy.
- **A/B testing** validated the improvement from augmentation.

### **4. Category Keyword Injection**
To prevent category confusion, predefined keywords were appended:

```python
category_keywords = {
    "Models of Manipulating Qubits for Computation": "Quantum gates, Qubit control, Quantum circuits, Quantum algorithms",
    "Methods of Building Qubits": "Superconducting qubits, Trapped ions, Quantum dots, Semiconductor qubits",
    "Address Obstacles to Quantum Computation": "Quantum error correction, Decoherence, Noise mitigation",
    "Applications of Quantum Computing": "Quantum simulation, Cryptography, Optimization problems"
}
