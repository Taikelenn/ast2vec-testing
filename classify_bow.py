import os
import tokenize
import time
import numpy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

correct_solutions_path = "data/correct/"
wrong_solutions_path = "data/wrong/"

embedding_times = []

cv = CountVectorizer()

def tokenize_python(text):
    t1 = time.perf_counter_ns()
    
    tokens = list(tokenize.generate_tokens(iter(text.split()).__next__))
    retval = [t.string for t in tokens]
    
    embedding_times.append(time.perf_counter_ns() - t1)
    
    return retval

# Converts a .py file to a map of tokens using bag of words
def file_to_vector(path):
    with open(path, "r") as f:
        code = f.read()
        return code

filenames = []
X = []
y = []

print("Reading from " + correct_solutions_path)
# Read correct solutions, label them "1"
for filename in os.listdir(correct_solutions_path):
    filenames.append(correct_solutions_path + filename)
    X.append(file_to_vector(correct_solutions_path + filename))
    y.append(1)

print("Reading from " + wrong_solutions_path)
# Read wrong solutions, label them "0"
for filename in os.listdir(wrong_solutions_path):
    filenames.append(wrong_solutions_path + filename)
    X.append(file_to_vector(wrong_solutions_path + filename))
    y.append(0)

# Run training as many times as k-fold is configured to & store results
precision_results, recall_results, f1_results = [], [], []
confmat_y_true, confmat_y_pred = [], []

kf = KFold(n_splits=5, shuffle=True)
for train_idx, val_idx in kf.split(X):
    X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
    X_val, y_val = [X[i] for i in val_idx], [y[i] for i in val_idx]
    
    cv = CountVectorizer(token_pattern=None, tokenizer=tokenize_python).fit(X_train)
    X_train_transformed = cv.transform(X_train).toarray()
    X_val_transformed = cv.transform(X_val).toarray()
    
    model = LogisticRegression().fit(X_train_transformed, y_train)
    y_val_predicted = model.predict(X_val_transformed)
    
    confmat_y_true.extend(y_val)
    confmat_y_pred.extend(y_val_predicted)
    
    precision_results.append(precision_score(y_val, y_val_predicted))
    recall_results.append(recall_score(y_val, y_val_predicted))
    f1_results.append(f1_score(y_val, y_val_predicted))

print(f"Avg embedding time: {(numpy.mean(embedding_times) / 1000000):.3f} ms")
print(f"Max embedding time: {(numpy.max(embedding_times) / 1000000):.3f} ms")

print(f"Precision: {numpy.mean(precision_results):.3f}")
print(f"Recall: {numpy.mean(recall_results):.3f}")
print(f"F1 score: {numpy.mean(f1_results):.3f}")

confmat = confusion_matrix(confmat_y_true, confmat_y_pred)
print(confmat)
