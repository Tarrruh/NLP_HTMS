import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import RBFSampler
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch
from google.colab import files

# Upload files interactively
uploaded = files.upload()

# Load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Drop rows with missing values
    data.dropna(subset=['Text', 'Class'], inplace=True)

    # Convert text to string type
    data['Text'] = data['Text'].astype(str)
    return data['Text'].tolist(), data['Class'].tolist()

# TF-IDF embeddings
def compute_tfidf_embeddings(texts, n_features=512):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_embeddings = tfidf_vectorizer.fit_transform(texts).toarray()

    rbf_sampler = RBFSampler(gamma=1.0, n_components=n_features, random_state=42)
    reduced_tfidf_embeddings = rbf_sampler.fit_transform(tfidf_embeddings)
    return reduced_tfidf_embeddings

# Optimized BERT embeddings
def compute_bert_embeddings(texts, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to('cuda')  # Use GPU

    embeddings = []
    model.eval()
    with torch.no_grad():
        # Ensure all inputs are strings
        texts = [str(text) for text in texts]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512).to('cuda')
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move to CPU
            embeddings.extend(batch_embeddings)

    return np.array(embeddings)

# Fuse TF-IDF and BERT embeddings
def fuse_embeddings(tfidf_embeddings, bert_embeddings):
    return np.hstack((tfidf_embeddings, bert_embeddings))

# Train and evaluate
def train_and_evaluate(file_path):
    texts, labels = load_data(file_path)

    print("Computing TF-IDF embeddings...")
    tfidf_embeddings = compute_tfidf_embeddings(texts)

    print("Computing BERT embeddings...")
    bert_embeddings = compute_bert_embeddings(texts)

    print("Fusing embeddings...")
    fused_embeddings = fuse_embeddings(tfidf_embeddings, bert_embeddings)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    labels = np.array(labels)

    for fold, (train_index, test_index) in enumerate(kf.split(fused_embeddings)):
        print(f"\nFold {fold + 1}")
        X_train, X_test = fused_embeddings[train_index], fused_embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        print(classification_report(y_test, predictions))

# Replace with uploaded file name
file_path = "data_train.csv"
train_and_evaluate(file_path)

# Prepare training embeddings
train_texts, train_labels = load_data(file_path)
print("Preparing training TF-IDF embeddings...")
train_tfidfs = compute_tfidf_embeddings(train_texts)

print("Preparing training BERT embeddings...")
train_bert = compute_bert_embeddings(train_texts)

train_embeddings = fuse_embeddings(train_tfidfs, train_bert)

# Upload and prepare test data
uploaded = files.upload()
test_file_path = "AWT_test_without_labels (1).csv"  # Replace with uploaded file name
test_data = pd.read_csv(test_file_path)
test_texts = test_data['Text'].tolist()

print("Preparing test TF-IDF embeddings...")
test_tfidfs = compute_tfidf_embeddings(test_texts)

print("Preparing test BERT embeddings...")
test_bert = compute_bert_embeddings(test_texts)

test_embeddings = fuse_embeddings(test_tfidfs, test_bert)

# Train and predict
print("Training Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(train_embeddings, train_labels)

print("Predicting on test data...")
test_predictions = model.predict(test_embeddings)

# Save predictions
output_file = "predictions3both.csv"
pd.DataFrame({'ID': range(1, len(test_predictions) + 1), 'Predicted Label': test_predictions}).to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
