import numpy as np
import random
import itertools
import pickle

class expression_tree:
    def __init__(self, operation, term, is_leaf):
        self.is_leaf = is_leaf
        self.operation = operation
        self.term = term
        self.left = None
        self.right = None

def random_tree_list(node_num, fail_max):
    # creates a random list of 1s, -1s that encodes a specific tree of fixed size.
    # there is a possibility of failure for each generation, as each list of 1s and -1s doesn't necessarily correspond to a valid tree
    # the probability of failure should be about 1/2, so repeating the operation up to fail_max times for fail_max even slightly large should guarantee success
    if node_num == 0:
        return [-1]
    fail_counter =0
    while(fail_counter < fail_max):
        fail = False
        tree_list = [1]
        one_counter = node_num-1
        minus_one_counter = node_num
        for i in range(2*node_num-1):
            p = one_counter / (one_counter+minus_one_counter)
            outcome = 2*np.random.binomial(1,p)-1
            if outcome == 1:
                one_counter -= 1
            else:
                minus_one_counter -= 1
            tree_list.append(outcome)
            if sum(tree_list) == -1:
                fail = True
                break
        if fail == True:
            fail_counter += 1
            continue
        tree_list.append(-1)
        return tree_list
    print("Error: failed to generate tree list")
    return None

def random_operator_list(node_num):
    # makes a list of node_num random operators
    #possible_operators = ['+', 'x', '//', '%']
    #possible_operators = ['+', '%']
    possible_operators = ['+', 'x', '//']
    operator_list = []
    for i in range(node_num):
        operator_list.append(random.choice(possible_operators))
    return operator_list

def random_term_list(node_num, is_recursive):
    # makes a list of node_num + 1 random terms. If is_recursive == True, then includes the recursive term r. Otherwise excludes it.
    if is_recursive:
        possible_terms = [1,2,'i', 'r']
    else:
        possible_terms = [2,'i']
    term_list = []
    for i in range(node_num+1):
        term_list.append(random.choice(possible_terms))
    return term_list


def lists_to_expression_tree(operator_list, term_list, tree_list):
    # takes a list of node_num operators, a list of node_num+1 terms, and a valid tree_list with node_num 1s and node_num+1 -1s and generates an expression_tree from it
    if tree_list.pop(0) == 1:
        root = expression_tree(operator_list.pop(0), None, False)
        root.left = lists_to_expression_tree(operator_list,term_list, tree_list)
        root.right = lists_to_expression_tree(operator_list,term_list, tree_list)
    else:
        root = expression_tree(None, term_list.pop(0), True)
    return root

def random_expression_tree(node_num, fail_max, is_recursive):
    # generates a random expression_tree by combining the above random generations
    return lists_to_expression_tree(random_operator_list(node_num), random_term_list(node_num, is_recursive), random_tree_list(node_num,fail_max))


def safe_int_div(a,b):
    # defines a "safe" integer division so that all expression_tree's give well defined functions
    if b != 0:
        return a//b
    else:
        return a

def safe_modulo(a,b):
    # defines a "safe" modulo operation so that all expression_tree's give well defined functions
    if b != 0:
        return a % b
    else:
        return a


def evaluate_expression_tree(exp_tree, i,r):
    # evaluates an expression_tree, using provided values of variables i and r
    if exp_tree.is_leaf == True:
        if exp_tree.term == "i":
            return i
        elif exp_tree.term == "r":
            return r
        else:
            return exp_tree.term
    else:
        if exp_tree.operation == "+":
            return evaluate_expression_tree(exp_tree.left,i,r)+evaluate_expression_tree(exp_tree.right,i,r)
        if exp_tree.operation == "x":
            return evaluate_expression_tree(exp_tree.left,i,r)*evaluate_expression_tree(exp_tree.right,i,r)
        if exp_tree.operation == "//":
            return safe_int_div(evaluate_expression_tree(exp_tree.left,i,r), evaluate_expression_tree(exp_tree.right,i,r))
        if exp_tree.operation == "%":
            return safe_modulo(evaluate_expression_tree(exp_tree.left,i,r), evaluate_expression_tree(exp_tree.right,i,r))

def parse_expression_tree(exp_tree):
    # parses an expression_tree into a mathematical expression
    if exp_tree.is_leaf == True:
        return str(exp_tree.term)
    else:
        return "("+parse_expression_tree(exp_tree.left) + ")"+exp_tree.operation +"("+ parse_expression_tree(exp_tree.right)+")"

def sequence_from_expression_tree(exp_tree, sequence_length, recursive_initial_val):
    # generates a sequence from an expression_tree, using the sequence index as i and the previous element in the sequence as the recursive element r, initialized as recursive_initial_val
    sequence_list = [evaluate_expression_tree(exp_tree, 0,recursive_initial_val)]
    for i in range(1,sequence_length):
        sequence_list.append(evaluate_expression_tree(exp_tree, i, sequence_list[-1]))
    return sequence_list


def random_sequence_with_parsing(node_num, fail_max, sequence_length, is_recursive, recursive_initial_val):
    exp_tree = random_expression_tree(node_num,fail_max, is_recursive)
    return parse_expression_tree(exp_tree), sequence_from_expression_tree(exp_tree, sequence_length,recursive_initial_val)

def random_sequence(node_num, fail_max, sequence_length, is_recursive, recursive_initial_val):
    return sequence_from_expression_tree(random_expression_tree(node_num,fail_max, is_recursive), sequence_length,recursive_initial_val)


def integer_list_to_binary_string(integer_list, binarized_sequence_length):
    binary_string = ""
    for i in integer_list:
        binary_string = binary_string + str(bin(i))[2:]
    if len(binary_string) < binarized_sequence_length:
        return None
    return binary_string[0:binarized_sequence_length]

def integer_list_to_integer_string(integer_list):
    integer_string = ""
    for i in integer_list:
        integer_string = integer_string +str(i) + ","
    return integer_string

def make_sequence_dict(max_node_num, sequence_length, binary_output = False):
    sequence_dict = {}
    for node_num in range(max_node_num+1):
        for tree_list in list(map(list,itertools.product(*([[-1,1]]*max(2*node_num-1,0))))):
            fail = False
            if node_num == 0:
                tree_list = tree_list+ [-1]
            else:
                tree_list = [1]+ tree_list + [-1]
            if sum(tree_list) != -1:
                continue
            for j in range(1,len(tree_list)):
                if sum(tree_list[0:j]) == -1:
                    fail = True
                    break
            if fail == True:
                continue
            for operator_list in list(map(list,itertools.product(*([['+', 'x', '//']]*node_num)))):
                for term_list in list(map(list,itertools.product(*([['i', 2]]*(node_num+1))))):
                    exp_tree = lists_to_expression_tree(operator_list[:], term_list[:], tree_list[:])
                    if binary_output == True:
                        output_string = integer_list_to_binary_string(sequence_from_expression_tree(exp_tree, sequence_length, sequence_length),sequence_length)
                    else:
                        output_string = str(sequence_from_expression_tree(exp_tree, sequence_length, sequence_length))
                    if sequence_dict.get(output_string) == None:
                        sequence_dict[output_string] = node_num
    inverse_dict = {}
    if binary_output == True:
        for k, v in sequence_dict.items():
            inverse_dict[v] = inverse_dict.get(v, []) + [k]
    else:
        for k, v in sequence_dict.items():
            inverse_dict[v] = inverse_dict.get(v, []) + [list(map(int, k.strip('][]').split(', ')))]
    return inverse_dict

def load_complexity_dict(max_complexity, sequence_length, binary_output = False):
    if binary_output == True:
        file_str = 'bin_c=' + str(max_complexity) + '_l=' + str(sequence_length) +'.pickle'
    else:
        file_str = 'int_c=' + str(max_complexity) + '_l=' + str(sequence_length) +'.pickle'
    try:
        dict_file = open(file_str, "rb")
    except:
        dict_file = open(file_str, "wb")
        pickle.dump(make_sequence_dict(max_complexity, sequence_length,binary_output), dict_file)
        dict_file.close()
        dict_file = open(file_str, "rb")
    return pickle.load(dict_file)

def random_sequence_of_complexity(complexity_list, max_complexity, sequence_length, binary_output = False):
    sequence_list = []
    if max(complexity_list) > max_complexity or min(complexity_list) < 0:
        print("complexity_list entries incompatible with max_complexity or negative")
        return None
    if binary_output == True:
        file_str = 'bin_c=' + str(max_complexity) + '_l=' + str(sequence_length) +'.pickle'
    else:
        file_str = 'int_c=' + str(max_complexity) + '_l=' + str(sequence_length) +'.pickle'
    try:
        dict_file = open(file_str, "rb")
    except:
        dict_file = open(file_str, "wb")
        pickle.dump(make_sequence_dict(max_complexity, sequence_length,binary_output), dict_file)
        dict_file.close()
        dict_file = open(file_str, "rb")
    complexity_dict = pickle.load(dict_file)
    for i in complexity_list:
        sequence_list.append(random.choice(complexity_dict[i]))
    # for i in range(max_complexity+1):
    #     print(len(complexity_dict[i]))
    return sequence_list


#print(load_complexity_dict(3,10))
