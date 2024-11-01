# Sentiment Analysis on Yelp Restaurant Reviews

## Introduction
This project is part of the Advanced Deep Learning (AIGC 5500) course, focusing on sentiment analysis using advanced NLP techniques. The primary objective is to classify Yelp restaurant reviews into positive, neutral, or negative sentiments by comparing the performance of two models: LSTM (Long Short-Term Memory) and DistilBERT (a distilled version of BERT).

## Project Structure

### Files Included
- `ADL_Final_Pro.ipynb`: This notebook contains the implementation of the LSTM model, including data preprocessing, model training, evaluation, and visualization of results.
- `Model_Distilbert1.ipynb`: This notebook includes the DistilBERT model implementation, fine-tuning, and evaluation.
- `ADL_Final_Report.docx`: The final report summarizing the project, including dataset description, model details, results, and conclusions.
- `Final_Project_ADL.pptx`: A PowerPoint presentation providing an overview of the project, including key findings and visualizations.

### Data Preprocessing
- **Text Cleaning:** Both notebooks begin with cleaning the text data, including removing HTML tags, punctuation, and converting text to lowercase.
- **Tokenization:** The text data is tokenized using appropriate tokenizers. For LSTM, standard tokenization is used, while DistilBERT utilizes `DistilBertTokenizer`.
- **Padding and Truncation:** Text sequences are padded and truncated to a maximum length of 200 tokens to ensure uniform input dimensions across all samples.

### Models
#### LSTM Model
- **Architecture:** The LSTM model consists of an embedding layer, two LSTM layers with 128 and 64 units respectively, followed by a dense layer and a dropout layer to prevent overfitting.
- **Training:** The model is trained using the Adam optimizer with categorical cross-entropy as the loss function. Training is done over 10 epochs with a batch size of 32.

#### DistilBERT Model
- **Architecture:** DistilBERT is a transformer-based model fine-tuned using the `TFDistilBertModel`. A custom classification layer is added on top of the pre-trained DistilBERT model.
- **Training:** Similar to the LSTM model, the DistilBERT model is trained for 10 epochs with a batch size of 32, fine-tuning the pre-trained layers for sentiment classification.

### Running the Notebooks
To run the notebooks:
1. Open `ADL_Final_Pro.ipynb` and run each cell sequentially. This notebook will preprocess the data, train the LSTM model, and evaluate its performance.
2. Open `Model_Distilbert1.ipynb` and run the cells sequentially. This notebook will handle the DistilBERT model, including tokenization, fine-tuning, and evaluation.

### Results
- The LSTM model outperformed DistilBERT in terms of accuracy, precision, recall, and F1-score.
  
  **LSTM Model Performance:**
  - **Accuracy:** 80.00%
  - **Precision:** 80.56%
  - **Recall:** 90.62%
  - **F1-score:** 85.29%

  **DistilBERT Model Performance:**
  - **Accuracy:** 64.00%
  - **Precision:** 40.96%
  - **Recall:** 64.00%
  - **F1-score:** 49.95%

### Visualizations
- Visualizations, including performance comparison plots, are provided within the notebooks. These graphs help illustrate the differences in performance between the two models.

### Conclusion
The LSTM model proved to be more effective for this specific sentiment classification task compared to DistilBERT. Further fine-tuning or different approaches might improve DistilBERT's performance, but in this setup, the LSTM model is the preferred solution.

### References
- TensorFlow: Used for building and training the LSTM model.
- Hugging Face Transformers: Used for fine-tuning and evaluating the DistilBERT model.
- Yelp Dataset: Provided the data for training and evaluation.
