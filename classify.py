import os
import ast
import ast2vec
import python_ast_utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

astmodel = ast2vec.load_model()

# Converts a .py file to a list of numbers using ast2vec
def file_to_vector(path):
    with open(path, "r") as f:
        code = f.read()
        tree = python_ast_utils.ast_to_tree(ast.parse(code))
        return astmodel.encode(tree).detach().tolist()

filenames = []
X = []
y = []

# Read correct solutions, label them "1"
for filename in os.listdir("data/correct"):
    filenames.append("data/correct/" + filename)
    X.append(file_to_vector("data/correct/" + filename))
    y.append(1)

# Read wrong solutions, label them "-1"
for filename in os.listdir("data/wrong"):
    filenames.append("data/wrong/" + filename)
    X.append(file_to_vector("data/wrong/" + filename))
    y.append(-1)

# Check for potential duplicates in embeddings
for idx, val in enumerate(X):
    for idx2, val2 in enumerate(X):
        if idx != idx2 and val == val2:
            print(f"Warning: same embedding for {filenames[idx]} and {filenames[idx2]}")

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25)
print(f"Training set size = {len(X_train)}, validation set size = {len(X_validation)}")

model = LogisticRegression().fit(X_train, y_train)
y_train_predicted = model.predict(X_train) # predict training data
print(f"Training accuracy {accuracy_score(y_train, y_train_predicted):.3f}") # compute accuracy on the training set

y_validation_predicted = model.predict(X_validation) # predict validation data
print(f"Validate accuracy {accuracy_score(y_validation, y_validation_predicted):.3f}") # compute accuracy on the validation set
