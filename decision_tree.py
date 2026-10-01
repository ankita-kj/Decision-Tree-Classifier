import numpy as np

# represents one node in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # node is a leaf if it contains a class prediction 
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    # calculate entropy or gini index
    def impurity(self, y):
        
        # find how many samples belong to each class 
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # calculate impurity using the selected criterion
        if self.criterion == "entropy":
            return -np.sum(probabilities * np.log2(probabilities))
        elif self.criterion == "gini":
            return 1 - np.sum(probabilities ** 2)
        else:
            raise ValueError("criterion must be 'entropy' or 'gini'")

    # calculate information gain for split
    def information_gain(self, y, y_left, y_right):
        parent_impurity = self.impurity(y)

        # calculate weight of each child
        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)

        # calculate weighted impurity of the children
        child_impurity = (
            left_weight * self.impurity(y_left)
            + right_weight * self.impurity(y_right)
        )

        return parent_impurity - child_impurity

    # return most common class label
    def majority_class(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    # find feature with the highest information gain
    def best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):

            # use median value of the feature as the threshold
            threshold = np.median(X[:, feature])

            # split data into left and right groups
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            # skip if the feature cannot split the data
            if not np.any(left_mask) or not np.any(right_mask):
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            gain = self.information_gain(y, y_left, y_right)

            # update if this split has the highest information gain
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_threshold

    # build the decision tree recursively
    def build_tree(self, X, y, depth):

        # stop if all data points have the same label
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        # stop if all data points have identical feature values
        if np.all(X == X[0]):
            return Node(value=self.majority_class(y))

        # stop if maximum depth has been reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self.majority_class(y))

        # best feature and threshold for next split
        feature, threshold = self.best_split(X, y)

        # if no valid split is found, create a leaf with the majority class
        if feature is None:
            return Node(value=self.majority_class(y))

        # split data using the best feature and threshold
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        # recursively build the left and right subtrees
        left_child = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)

        # create a node with the chosen split and its children
        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    # train decision tree
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.root = self.build_tree(X, y, 0)

        return self

    # predict class label of one data point
    def predict_one(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)

        return self.predict_one(x, node.right)

    # predict class labels for all data points
    def predict(self, X):
        X = np.asarray(X)

        predictions = []

        for x in X:
            prediction = self.predict_one(x, self.root)
            predictions.append(prediction)

        return np.array(predictions)