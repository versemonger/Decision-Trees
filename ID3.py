import numpy as np
import math
import argparse as ap
from Data import Data
from Node import Node


def p_rate(target):
    """

    :param target: target array
    :return: poisonous rate
    """
    return np.sum(target == 'p') / float(len(target))


def entropy(target):
    """

    :param target: an np array
    :return: entropy of target array
    """
    length = len(target)
    if length == 0:
        raise RuntimeError('Target array is empty!')
    p = p_rate(target)
    if p == 1 or p == 0:
        return 0
    return -(math.log(p, 2) * p + math.log(1 - p, 2) * (1 - p))


def mis_cla_single(target):
    """

    :param target: an np array
    :return: mis classification error rate of target array
    """
    length = len(target)
    if length == 0:
        raise RuntimeError('Target array is empty!')
    p = p_rate(target)
    if p <= 0.5:
        return p
    else:
        return 1 - p


def mis_classification_rate(attri_index, obj_indices):
    """
    :param attri_index: index of the attribute whose purity function
                        will be calculated. Range: (1 - 22)
    :param obj_indices: (np array) indices of objects of data set in current node
    :return: value of purity
    """
    if len(obj_indices) == 0:
        raise RuntimeError('This attribute is empty!')
    else:
        target = Data.train[obj_indices, 0]
        total_error_rate = mis_cla_single(target)
        value_set = Data.attri_option_list[attri_index - 1]  # Set of values belonging to this attribute
        value_num = len(value_set)
        obj1 = Data.train[obj_indices, attri_index]  # Set of objects
        obj_num = np.zeros(value_num, dtype=int)  # Number of objects in each category
        sub_error_rate = np.zeros(value_num, dtype=float)  # Error rate in each category
        for i in range(value_num):  # Calculate error rate of each category of objects
            value = value_set[i]
            subset = obj1 == value
            obj_num[i] = np.sum(subset)
            if obj_num[i] == 0:
                sub_error_rate[i] = 0
            else:
                sub_error_rate[i] = mis_cla_single(target[subset])
        obj_total_num = np.sum(obj_num)
        sub_error_sum = 0
        for i in range(value_num):  # Calculate total entropy after splitting
            sub_error_sum += sub_error_rate[i] * obj_num[i] / obj_total_num
        return total_error_rate - sub_error_sum


def gain_info_single(attri_index, obj_indices):
    """
    :param attri_index: index of the attribute whose info gain
                        will be calculated. Range: (1 - 22)
    :param obj_indices: (np array) indices of objects of data set in current node
    :return: value of info gain
    """
    if len(obj_indices) == 0:
        raise RuntimeError('This attribute is empty!')
    else:
        target = Data.train[obj_indices, 0]
        total_entropy = entropy(target)
        value_set = Data.attri_option_list[attri_index - 1]  # Set of values belonging to this attribute
        value_num = len(value_set)
        obj1 = Data.train[obj_indices, attri_index]  # Set of objects
        obj_num = np.zeros(value_num, dtype=int)  # Number of objects in each category
        sub_entropy = np.zeros(value_num, dtype=float)  # Entropy in each category
        for i in range(value_num):  # Calculate entropy of each category of objects
            value = value_set[i]
            subset = obj1 == value
            obj_num[i] = np.sum(subset)
            if obj_num[i] == 0:
                sub_entropy[i] = 0
            else:
                sub_entropy[i] = entropy(target[subset])
        obj_total_num = np.sum(obj_num)
        sub_entropy_sum = 0
        for i in range(value_num):  # Calculate total entropy after splitting
            sub_entropy_sum += sub_entropy[i] * obj_num[i] / obj_total_num
        # print total_entropy - sub_entropy_sum
        return total_entropy - sub_entropy_sum


def gain_info_select(attri_indices, obj_indices):
    """

    :param attri_indices: Set of attributes in this data subset.
    :param obj_indices: Set of objects in this data subset.
    :return: The attribute with maximum info gain
    """
    if len(attri_indices) == 0 or len(obj_indices) == 0:
        raise RuntimeError("No attribute or no object")
    else:
        max_info_gain = 0
        max_info_index = 0
        for i in attri_indices:  # Calculate info gain of each attribute and find the best attribute
            if Data.mode == 'i':
                current_gain = gain_info_single(i, np.array(obj_indices))
            else:
                current_gain = mis_classification_rate(i, np.array(obj_indices))
            if current_gain > max_info_gain:
                max_info_gain = current_gain
                max_info_index = i
        return max_info_index


def delete_index(attri_indices, selected_index):
    """

    :param attri_indices: set of attributes before splitting
    :param selected_index: selected index of this node
    :return: set of attributes after splitting
    """
    return [index0 for index0 in attri_indices if index0 != selected_index]


def split_obj(selected_index, obj_indices, option):
    """

    :param selected_index: referring to attribute used to split the objects
    :param obj_indices: data set of current node
    :param option: decide which set of data you are splitting
    :return: split data sets
    """
    if len(obj_indices) == 0:
        raise RuntimeError("No objects")
    value_set = Data.attri_option_list[selected_index - 1]
    obj_subset0 = []
    data_set_type = {'train': Data.train, 'test': Data.test,
                     'validation': Data.validation}
    data_set = data_set_type[option]
    for value in value_set:  # Split objects into different sets according their value of the attribute
        obj_subset0.append(obj_indices[value == data_set[obj_indices, selected_index]])
    return obj_subset0


def find_majority(target):
    """

    :param target: an array of target
    :return: 'p' or 'e', depending on which appears more in target
    """
    p = p_rate(target)
    if p >= 0.5:
        return 'p'
    else:
        return 'e'


def build_dt(attri_indices, obj_indices):
    """

    :param attri_indices: Set of indices of attributes for this subtree.
    :param obj_indices: Set of indices of objects for this subtree.
    :return: Root of this decision tree constructed from data as stated above.
    """
    if len(obj_indices) == 0:
        raise RuntimeError('No objects in this node')
    else:
        node = Node()
        # If all attributes have been used in this path make this node a leaf node.
        if len(attri_indices) == 0:
            node.result = find_majority(Data.train[obj_indices, 0])
            return node
        # If target in this subset of data is pure, make this node a leaf node.
        elif entropy(Data.train[obj_indices, 0]) == 0:
            node.result = Data.train[obj_indices[0]][0]
            return node
        # Find the best attribute for splitting and split the data set with that attribute
        else:
            if len(attri_indices) == 1:
                selected_index = attri_indices[0]
            else:
                selected_index = gain_info_select(attri_indices, obj_indices)
            node.index = selected_index
            subsets = split_obj(selected_index, obj_indices, 'train')
            if not chi_square_test(obj_indices, subsets):
                node.result = find_majority(Data.train[obj_indices, 0])
                return node
            attri_subset = delete_index(attri_indices, selected_index)
            # Build subtrees with subsets of data
            for subset in subsets:
                # Make the empty subset corresponding to some attribute a leaf node
                if len(subset) == 0:
                    new_child = Node()
                    new_child.result = find_majority(Data.train[obj_indices, 0])
                    node.child.append(new_child)
                else:
                    node.child.append(build_dt(attri_subset, subset))
            return node


def list_decision_tree(tree_list, root, depth):
    """

    :param tree_list: list of the decision tree of the form [[level1 nodes], [level2 nodes], ...]
    :param root: Root of decision tree
    :param depth: Depth of root node in the whole decision tree, starting from 1 here
    :return: A list of decision tree where nodes are separated by level
    """
    if len(tree_list) < depth:
        tree_list.append([root])
    else:
        tree_list[depth - 1].append(root)
    for sub_tree in root.child:
        list_decision_tree(tree_list, sub_tree, depth + 1)


def display_decision_tree(dc_tree_list):
    """

    :param dc_tree_list: A list of decision tree where nodes are separated by level
    :return: none
    """
    for tree_ls in dc_tree_list:
        for node in tree_ls:
            if node.result != '?':
                print node.result,
            else:
                print node.index, "(Num of children: ", len(node.child), ")",
        print


def classify(tree, obj_indices, result):
    """

    :param tree: decision tree
    :param obj_indices: indices of objects to be classified
    :param result: classification result of all objects
    :return:
    """
    if len(obj_indices) == 0:
        pass
    if tree.result != '?':
        result[obj_indices] = tree.result
    else:
        if Data.validation_flag:
            subsets = split_obj(tree.index, obj_indices, 'validation')
        else:
            subsets = split_obj(tree.index, obj_indices, 'test')
        num_subset = len(subsets)
        for i in range(num_subset):
            classify(tree.child[i], subsets[i], result)


def chi_square_test(obj_indices, subsets):
    """

    :param obj_indices: Set of objects before splitting
    :param subsets: Sets of objects after splitting
    :return: Whether the splitting passes chi square test, True for yes and False for no
    """
    freedom = len(subsets) - 1
    if freedom <= 0:
        raise RuntimeError("Freedom should be at least 1")
    target = Data.train[obj_indices, 0]
    c = 0
    p = p_rate(target)  # Calculate the portion of poisonous mushrooms
    n = 1 - p
    for subset in subsets:
        length = len(subset)
        if length == 0:
            continue
        else:
            p_estimate = length * p
            n_estimate = length * n
            sub_target = Data.train[subset, 0]
            p_actual = np.sum(sub_target == 'p')
            n_actual = length - p_actual
            d1 = p_estimate - p_actual
            d2 = n_estimate - n_actual
            c += d1 * d1 / p_estimate + d2 * d2 / n_estimate
    if Data.conf_level == 3:
        threshold = 0
    else:
        threshold = Data.chi_table[freedom - 1][Data.conf_level]
    if c >= threshold:
        return True
    else:
        return False


def main():
    Data()  # Initialize data set
    attribute_indices = np.arange(1, 23)
    obj = np.arange(4062)
    tree = build_dt(attribute_indices, obj)
    if Data.display_tree_flag:
        tree_list_by_level = []
        list_decision_tree(tree_list_by_level, tree, 1)
        display_decision_tree(tree_list_by_level)
    test_num = 2031
    result = np.chararray(test_num)
    test_obj = np.arange(test_num)
    classify(tree, test_obj, result)
    if Data.mode == 'i':
        print "Entropy,",
    else:
        print "Misclassification Error,",
    print "Confidence Level: {}".format(args.confidence_level),
    if not Data.validation_flag:
        print "Accuracy: {}".format(np.sum(result == Data.test[test_obj, 0]) / float(test_num))
    else:
        print "Result printed in validation_result.txt."
        f = open('validation_result.txt', 'w')
        for x in result:
            print >>f, x
        f.close()


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-m", "--mis_classify_error", action="store_true",
                        help="Use mis_classification error rate as criterion"
                        "for splitting, info gain method is default method")
    parser.add_argument("-c", "--confidence_level", type=int, choices=[0, 50, 95, 99],
                        default=95, help="Specify the confidence level")
    parser.add_argument("-v", "--validation", action="store_true",
                        help="Validate the model with validation data set")
    parser.add_argument("-d", "--display_tree", action="store_true",
                        help="Display the decision tree.")
    args = parser.parse_args()
    if args.mis_classify_error:
        Data.mode = 'm'
    if args.validation:
        Data.validation_flag = True
    if args.display_tree:
        Data.display_tree_flag = True
    level_map = {50: 0, 95: 1, 99: 2, 0: 3}
    Data.conf_level = level_map[args.confidence_level]
    main()



