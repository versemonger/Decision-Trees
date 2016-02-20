class Node:
    """
    Each Node object represents a node in decision tree.
    """
    def __init__(self):
        self.index = 0  # Index of splitting attribute.
        self.child = []
        self.result = '?'  # 'p' for poisonous, 'e' for edible, '?' for not leaf node.

    def add_child(self, child):
        self.child.append(child)

    def add_result(self, result):
        self.result = result
