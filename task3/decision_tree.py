import numpy as np


class MyDecisionTree:

    def __init__(self, max_deep, min_size):
        self.max_deep = max_deep
        self.min_size = min_size
        self.root = None
        self.classes = None

    @staticmethod
    def gini_index(groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0

        for group in groups:

            size = float(len(group))
            # print('size = ', group.size)
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)

        return gini

    @staticmethod
    def test_split(index, value, dataset):
        left, right = list(), list()

        for row in dataset:

            if row[index] < value:
                left.append(row)
            else:
                right.append(row)

        return left, right

    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 0, 0, 1, None

        for index in range(dataset[0].size - 1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups

        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    @staticmethod
    def to_terminal(group):
        outcomes = [row[-1] for row in group]

        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, depth):
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_deep:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:

            node['left'] = self.get_split(left)


            self.split(node['left'], depth + 1)

        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:

            node['right'] = self.get_split(right)

            self.split(node['right'], depth + 1)

    # Build a decision tree
    def fit_data(self, train_data, target_labels):
        self.classes = np.unique(target_labels)
        train = np.column_stack((train_data, target_labels))
        self.root = self.get_split(train)
        # print('init root = ', self.root)
        self.split(self.root, 1)
        return self.root

    def predict_column(self, node, column):
        if column[self.root['index']] <= self.root['value']:
            node = node['left']
            if node == self.classes[0]:
                return node
            node['left'] = self.predict_column(node, column)
            return node['left']

        else:
            node = node['right']
            if node == self.classes[1]:
                # print('1 - ', node == self.classes[1])
                return node
            # print('2 - ', node)
            node['right'] = self.predict_column(node, column)
            return node['right']

    def predict_one_column(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_one_column(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_one_column(node['right'], row)
            else:
                return node['right']

    def predict_data(self, data):
        classes = list()
        for column in data:

            classes.append(self.predict_one_column(self.root, column))

        return np.array(classes)


# a = MyDecisionTree(1, 1)
#
# dataset1 = np.array([[2.771244718, 1.784783929],
#                      [1.728571309, 1.169761413],
#                      [3.678319846, 2.81281357],
#                      [3.961043357, 2.61995032],
#                      [2.999208922, 2.209014212],
#                      [7.497545867, 3.162953546],
#                      [9.00220326, 3.339047188],
#                      [7.444542326, 0.476683375],
#                      [10.12493903, 3.234550982],
#                      [6.642287351, 3.319983761]])
#
# classes1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
#
# dataset2 = np.array([[2.771244718, 1.784783929],
#                      [6.642287351, 3.319983761]])
# classes2 = np.array([0, 1])
#
# print(a.fit_data(dataset1, classes1))
#
# print(a.predict_data(np.array([[5, 2], [10, 3]])))
