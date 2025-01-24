from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data['Text'], data['Class']

# Compute TF-IDF embeddings
def compute_tfidf_embeddings(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    return vectorizer.fit_transform(texts).toarray(), vectorizer

# Train and evaluate with cross-validation
def train_and_evaluate(file_path):
    texts, labels = load_data(file_path)
    embeddings, vectorizer = compute_tfidf_embeddings(texts)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
    labels = np.array(labels)

    all_predictions = []
    for fold, (train_index, test_index) in enumerate(skf.split(embeddings, labels)):
        print(f"\nFold {fold + 1}")
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        classifier = LogisticRegression(max_iter=1000, random_state=40)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        all_predictions.extend(predictions)
        print(classification_report(y_test, predictions))
    
    return vectorizer, classifier

# Prepare prediction dataset
def predict_and_save(file_path, vectorizer, classifier, output_file):
    test_data = pd.read_csv(file_path)
    test_texts = test_data['Text']

    test_embeddings = vectorizer.transform(test_texts).toarray()
    test_predictions = classifier.predict(test_embeddings)

    output_df = pd.DataFrame({'ID': range(1, len(test_predictions) + 1), 'Predicted Label': test_predictions})
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# File paths
train_file_path = 'saB_data.csv'  # Training data file
test_file_path = 'M_Test.csv'       # Test data file
output_file = 'predictions_malayalam_tfidf.csv'  # Output predictions file

# Train and generate predictions
vectorizer, classifier = train_and_evaluate(train_file_path)
predict_and_save(test_file_path, vectorizer, classifier, output_file)