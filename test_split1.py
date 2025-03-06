from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

test_sizes = [0.1, 0.25, 0.5, 0.75, 0.9]
accuracy_means = []
accuracy_std = []

for test_size in test_sizes:
    scores = []
    for random_state in range(10):  # Run multiple times to get an average
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        scores.append(accuracy_score(y_test, y_pred))

    accuracy_means.append(np.mean(scores))
    accuracy_std.append(np.std(scores))

plt.figure(figsize=(8, 5))
plt.errorbar(test_sizes, accuracy_means, yerr=accuracy_std, fmt='o-', capsize=5, label="Accuracy")
plt.xlabel("Test Split Ratio")
plt.ylabel("Accuracy Score")
plt.title("Decision Tree Accuracy vs. Test Split Ratio")
plt.legend()
plt.grid()
plt.show()
