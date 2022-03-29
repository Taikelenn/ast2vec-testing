import os
import ast
import ast2vec
import python_ast_utils
import time
import numpy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

correct_solutions_path = "data/correct/"
wrong_solutions_path = "data/wrong/"

astmodel = ast2vec.load_model()

embedding_times = []

# Converts a .py file to a list of numbers using ast2vec
def file_to_vector(path):
    with open(path, "r") as f:
        code = f.read()
        
        t1 = time.perf_counter_ns()
        
        # This part is timed
        tree = python_ast_utils.ast_to_tree(ast.parse(code))
        result_tensor = astmodel.encode(tree)
        result = result_tensor.detach().tolist()
        
        embedding_times.append(time.perf_counter_ns() - t1)
        
        # Decode the embeddings; see thesis, 4.3.
        tree_decoded = astmodel.decode(result_tensor, max_size=300)
        with open(path.replace("data/", "data_decoded/"), "w") as f_out:
            f_out.write(str(tree))
            f_out.write("\n")
            f_out.write(str(tree_decoded))

        return result

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

# Check for potential duplicates in embeddings
for idx, val in enumerate(X):
    for idx2, val2 in enumerate(X):
        if idx != idx2 and val == val2:
            print(f"Warning: same embedding for {filenames[idx]} and {filenames[idx2]}")

# Run training as many times as k-fold is configured to & store results
precision_results, recall_results, f1_results = [], [], []
confmat_y_true, confmat_y_pred = [], []

kf = KFold(n_splits=5, shuffle=True)
for train_idx, val_idx in kf.split(X):
    X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
    X_val, y_val = [X[i] for i in val_idx], [y[i] for i in val_idx]
    
    model = LogisticRegression().fit(X_train, y_train)
    y_val_predicted = model.predict(X_val)
    
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
