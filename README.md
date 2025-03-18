# Explainable AI for Adverse Vaccine Reactions

## Overview
Machine Learning has made significant progress in recent years, leading to highly complex models that achieve remarkable accuracy. However, these "black-box" models lack transparency, making it difficult to understand how they make predictions. This is especially critical in fields like healthcare, where interpretability is essential for trust, accountability, and fairness.

This project contributes to the field of Explainable AI (XAI) by improving the interpretability of machine learning models used to predict adverse reactions to COVID-19 vaccines. Specifically, it investigates how incorporating external background knowledge, such as environmental data, enhances model transparency and provides deeper insights into prediction factors.

## Domain and Dataset
The focus of this project is on the medical domain, particularly the side effects of COVID-19 vaccinations. Among the various reported side effects, this study analyzes headaches, a commonly reported symptom post-vaccination. The aim is to explore how medical, demographic, and environmental factors contribute to vaccine-related headaches.

The dataset used for this project was sourced from Kaggle and consists of three relational files containing:
- Patient demographic data
- Vaccine administration records
- Reported side effects

Significant preprocessing was performed to clean and refine the dataset by handling missing values, standardizing formats, and addressing potential biases.

## Dataset Enrichment
To enhance prediction accuracy, external data on air and water pollution levels from various U.S. states were integrated. This additional background knowledge provided environmental context, potentially influencing vaccine side effects. A temporal alignment ensured that pollution records matched the year of the patient data, maintaining consistency across data sources.

### Text Embedding Generation
The dataset contained textual data, such as patient history and allergies, which were transformed into numerical embeddings using the BioClinicalBERT pre-trained language model. The text was tokenized, and embeddings were extracted from the model’s last hidden state. These embeddings were averaged and incorporated into the dataset. Principal Component Analysis (PCA) was applied to reduce dimensionality while retaining significant information.

## Model Training and Evaluation
Various machine learning models were trained and evaluated, including both white-box and black-box approaches. The best-performing model was a hybrid approach combining:
- A **Random Forest classifier**
- A **deep learning model**

This model achieved an accuracy of **64.42%**, demonstrating the potential of hybrid modeling techniques in improving both accuracy and interpretability.

## Key Findings
- **Domain knowledge improves interpretability**: Incorporating pollution-related features led to an increase in model accuracy, indicating that environmental factors contribute to predicting adverse reactions.
- **Limited impact of pollution**: While atmospheric pollution was found to influence vaccine side effects, its effect was not substantial, suggesting that other factors (e.g., genetic, medical history) play a more dominant role.
- **Feature importance**: Factors such as time of vaccination and environmental conditions were among the most influential in model predictions.

## Future Work
To build on these findings, future research could:
- Expand the dataset to include additional adverse reactions
- Test alternative text embedding methods
- Optimize hybrid model architectures
- Integrate additional contextual factors, such as social or economic influences, to enhance predictions

---
This project contributes to the growing field of **Explainable AI** by demonstrating how external domain knowledge can improve the interpretability of machine learning models in critical healthcare applications.

