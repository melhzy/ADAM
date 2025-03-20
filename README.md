# ADAM-1: An AI Reasoning and Bioinformatics Model for Alzheimer’s Detection and Microbiome-Clinical Data Integration

Alzheimer's Disease Analysis Model (ADAM) is a comprehensive framework designed to enhance our understanding of Alzheimer's Disease (AD) through the integration of advanced data analysis, machine learning, and reasoning techniques. By leveraging multi-agent systems, ADAM-1, as the first generation of this agentic AI system, aims to process complex datasets, summarize critical information, and classify findings to support research and clinical decision-making.

## Architecture

The ADAM-1 framework comprises three primary agents:
<p align="center">
  <img src="https://github.com/user-attachments/assets/8d0ac0ea-a159-427a-bf62-c9db84f78260" alt="ADAM-1 Architecture" width="80%">
</p>

1. **Computation Agent**: This agent focuses on machine learning and computational biology to process and analyze complex datasets. It employs advanced algorithms to identify patterns and correlations within the data, contributing to the detection and understanding of AD.

2. **Summarization Agent**: This agent utilizes reasoning logic to interpret data processed by the Computation Agent. Through a Chain of Thoughts design, it generates coherent and concise summaries of the findings, facilitating easier comprehension and further analysis.

3. **Classification Agent**: Building upon the summaries provided by the Summarization Agent, this agent applies reasoning logic to categorize the information. Using the Chain of Thoughts approach, it ensures that classifications are logical and transparent, aiding in accurate diagnosis and treatment planning.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f5a2076d-139a-4f34-a949-4f2a9d176c52" alt=" Agentic Systems Architecture" width="80%">
</p>

The interaction among these agents ensures a seamless flow from data analysis to summarization and classification, enhancing the overall understanding of Alzheimer's Disease.

## Key Features

- **Multi-Modal Data Integration**: ADAM-1 integrates diverse data sources, including genetic, clinical, and imaging data, to provide a holistic view of Alzheimer's Disease.

- **Retrieval-Augmented Generation (RAG)**: By combining retrieval systems with generative models, ADAM-1 ensures that the most relevant and accurate information is utilized in analyses.

- **Robust Performance**: The framework is designed for scalability and efficiency, capable of handling large datasets and complex computations without compromising performance.

## Visual Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/39a89a62-29bb-42bc-a4f2-8583bc36ab43" alt="Comparative Analysis of XGBoost and ADAM" width="80%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b1eb170c-f32d-4d01-a0a6-c8e3488d6559" alt="Performance Comparison of F1 Scores Between XGBoost and ADAM" width="80%">
</p>


## Hardware and Software

The experiments were conducted on an Ubuntu 24.04.2 LTS workstation equipped with an Intel® Core™ i9-10900X (20 threads), 128 GB RAM, and four NVIDIA GeForce RTX™ 3090 GPUs, providing a robust computing environment for LLM and machine learning tasks. The software stack was built on Python 3.10.14, utilizing XGBoost 2.1.3, and Scikit-Learn 1.5.2 for model training, with Optuna 4.1.0 handling hyperparameter optimization. For LLM processing, the system integrated OpenAI 1.55.1, PandasAI 2.4.2, LangChain 0.3.8, and LangChain-Chroma 0.1.4. Additionally, Scikit-Bio 0.6.2, SciPy 1.10.1, NumPy 1.26.4, and Pandas 1.5.3 facilitated data processing and analysis. For visualization and interpretability, the setup included Matplotlib 3.7.5, Seaborn 0.12.2, and SHAP 0.46.0. This configuration ensured high computational efficiency and scalability for deep learning workflows.

## Future Directions

Future iterations of ADAM-1 aim to incorporate additional data modalities, such as neuroimaging and biomarkers, to broaden its scalability and applicability for Alzheimer's research and diagnostics.

## Conclusion

ADAM-1 represents a significant advancement in the integration of AI and bioinformatics for Alzheimer's disease detection and analysis. Its multi-agent architecture and RAG techniques provide a robust framework for synthesizing insights from diverse data sources, offering a promising tool for researchers and clinicians in the field of neurodegenerative diseases.

For more detailed information, please refer to the full paper: 
https://doi.org/10.48550/arXiv.2501.08324
