import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch
from google.colab import files
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    if 'Text' not in data.columns or 'Class' not in data.columns:
        raise ValueError("CSV file must contain 'Text' and 'Class' columns.")
    texts = data['Text'].fillna("").astype(str).tolist()  # Replace NaN with empty strings
    labels = data['Class']
    return texts, labels
# Compute BERT embeddings on GPU
def compute_bert_embeddings(texts, batch_size=32):
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("Input to BERT tokenizer must be a list of strings.")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)  # Load model to GPU
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            print(f"Processing batch {i // batch_size + 1} out of {len(texts) // batch_size + 1}...")
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move embeddings to CPU
            embeddings.extend(batch_embeddings)
    return np.array(embeddings)
# Train and evaluate
def train_and_evaluate(file_path):
    texts, labels = load_data(file_path)
    print("Computing BERT embeddings...")
    bert_embeddings = compute_bert_embeddings(texts)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    labels = np.array(labels)
    for fold, (train_index, test_index) in enumerate(kf.split(bert_embeddings)):
        print(f"\nFold {fold + 1}")
        X_train, X_test = bert_embeddings[train_index], bert_embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        print(classification_report(y_test, predictions))
# Upload training data
print("Please upload the training data CSV file:")
uploaded = files.upload()
train_file_path = list(uploaded.keys())[0]
# Train and evaluate model
train_and_evaluate(train_file_path)
# Prepare test data
print("Please upload the test data CSV file:")
uploaded = files.upload()
test_file_path = list(uploaded.keys())[0]
test_data = pd.read_csv(test_file_path)
test_texts = test_data['Text'].fillna("").astype(str).tolist()
# Train on full training data and make predictions
train_texts, train_labels = load_data(train_file_path)
print("Computing BERT embeddings for training data...")
train_bert = compute_bert_embeddings(train_texts)
print("Computing BERT embeddings for test data...")
test_bert = compute_bert_embeddings(test_texts)
model = RandomForestClassifier(random_state=42)
model.fit(train_bert, train_labels)
test_predictions = model.predict(test_bert)
# Save predictions to CSV
output_file = f"predictions_{train_file_path.split('.')[0]}_bert.csv"
pd.DataFrame({'ID': range(1, len(test_predictions) + 1), 'Predicted Label': test_predictions}).to_csv(output_file, index=False)
# Download predictions
print(f"Predictions saved to {output_file}")
files.download(output_file)
