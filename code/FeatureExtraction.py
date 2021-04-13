import os
import itertools
import numpy as np
from itertools import product
from numpy import random
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec

random_seed = 40


def combine_seq_file(seq_files, target_dir):
    suffix = os.path.splitext(seq_files[0])[-1]
    return target_dir + '/' + 'combined_input_file' + suffix


def count_num(category, seq_file, label_file, output_file):
    if category == 'DNA':
        alphabet = "ATCG"
    elif category == 'RNA':
        alphabet = "AUCG"
    elif category == "Protein":
        alphabet = "ACDEFGHIKLMNPQRSTVWY"

    num_list = []
    len_list = []
    detail_list = []

    for i in range(len(seq_file)):
        with open(seq_file[i], 'r') as f:
            flag = 0
            temp_detail_list = []
            for line in f.readlines():
                line = line.strip('\n')
                if line[0] == '>' and flag == 0:
                    temp = ""
                    continue
                elif line[0] == '>' and flag == 1:
                    flag = 0
                else:
                    temp += line
                    flag = 1
                    continue
                temp_detail_list.append(temp)
                len_list.append(len(temp))
                temp = ""
            detail_list.append(temp_detail_list)
            num_list.append(len(temp_detail_list))

    with open(output_file, 'w') as f:
        for i in range(len(label_file)):
            for j in range(len(detail_list[i])):
                f.write('>Sequence[' + str(j+1) + '] | Label[' + str(i) + ']\n')
                f.write(detail_list[i][j])
            f.write('\n')

    return num_list, len_list


def generate_label_list(num_list, label):
    label_list = []
    for i in range(len(label)):
        label_list += [int(label[i])] * num_list[i]
    return np.array(label_list)


def possible_parameter_generation(args, parameter_dict):
    if args.method == 'SVM':
        possible_parameter_generation_svm(args.cost, args.gamma, parameter_dict)
    elif args.method == 'RF':
        possible_parameter_generation_rf(args.tree, parameter_dict)
    elif args.method == 'KNN':
        possible_parameter_generation_knn(args.ngb, parameter_dict)
    elif args.method == 'LinearSVM':
        possible_parameter_generation_lsvm(args.cost, parameter_dict)
    return parameter_dict


def possible_parameter_generation_lsvm(cost, parameter_dict):
    if cost is not None:
        if len(cost)==1:
            range_of_cost = range(cost[0], cost[0]+1, 1)
        elif len(cost)==2:
            range_of_cost = range(cost[0], cost[1], 1)
        elif len(cost)==3:
            range_of_cost = range(cost[0], cost[1], cost[2])
    else:
        range_of_cost = range(-10, 11, 1)
    parameter_dict['cost'] = range_of_cost


def possible_parameter_generation_knn(ngb, parameter_dict):
    if ngb is not None:
        if len(ngb) == 1:
            range_of_ngb = range(ngb[0], ngb[0] + 1, 1)
        elif len(ngb) == 2:
            range_of_ngb = range(ngb[0], ngb[1], 1)
        elif len(ngb) == 3:
            range_of_ngb = range(ngb[0], ngb[1], ngb[2])
    else:
        range_of_ngb = range(1, 20, 1)
    parameter_dict['ngb'] = range_of_ngb
    return parameter_dict


def possible_parameter_generation_svm(cost, gamma, parameter_dict):
    if cost is not None:
        if len(cost)==1:
            range_of_cost = range(cost[0], cost[0]+1, 1)
        elif len(cost)==2:
            range_of_cost = range(cost[0], cost[1], 1)
        elif len(cost)==3:
            range_of_cost = range(cost[0], cost[1], cost[2])
    else:
        range_of_cost = range(-5, 11, 3)

    if gamma is not None:
        if len(gamma) == 1:
            range_of_gamma = range(gamma[0], gamma[0]+1, 1)
        elif len(gamma) == 2:
            range_of_gamma = range(gamma[0], gamma[1], 1)
        elif len(gamma) == 3:
            range_of_gamma = range(gamma[0], gamma[1], gamma[2])
    else:
        range_of_gamma = range(-10, 6, 3)

    parameter_dict['cost'] = list(range_of_cost)
    parameter_dict['gamma'] = list(range_of_gamma)
    return parameter_dict


def possible_parameter_generation_rf(tree, parameter_dict):
    if tree is not None:
        if len(tree) == 1:
            range_of_tree = range(tree[0], tree[0]+1, 1)
        elif len(tree) == 2:
            range_of_tree = range(tree[0], tree[1], 1)
        elif len(tree) == 3:
            range_of_tree = range(tree[0], tree[1], tree[2])
    else:
        range_of_tree = range(10, 200, 10)

    parameter_dict['tree'] = list(range_of_tree)
    return parameter_dict


def dict_to_list(dict):
    parameter_list = []
    key_list = list(dict.keys())
    for value_pair in product(*list(dict.values())):
        temp = {}
        for i in range(len(value_pair)):
            temp[key_list[i]] = value_pair[i]
        parameter_list.append(temp)
    return parameter_list


def divide_data_set(args, label_list):
    x = random.normal(loc=0.0, scale=1, size=len(label_list))
    num_of_folds = int(args.test)
    folder = StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=random.RandomState(random_seed))
    fold_result = list(folder.split(x, label_list))
    args.folds = fold_result
    return args


def encode_line_onehot(args, line):
    if args.type == 'DNA':
        alphabet = "ATCG"
    elif args.type == 'RNA':
        alphabet = "AUCG"
    elif args.type == "Protein":
        alphabet = "ACDEFGHIKLMNPQRSTVWY"

    result = []
    line = line.strip("\n")
    # print(line)
    for c in line:
        # print('1'+c+'1')
        num = alphabet.index(c)
        vector = [0] * len(alphabet)
        vector[num] = 1
        result.append(vector)
    return result


def generate_kmer_list(k, alphabet):
    return ["".join(i) for i in itertools.product(alphabet, repeat=k[0])]


def kmer_frequency_count(kmer, line):
    i = 0
    j = 0
    count = 0
    len_line = len(line)
    len_kmer = len(kmer)
    while i < len_line and j < len_kmer:
        if line[i] == kmer[j]:
            i += 1
            j += 1
            if j >= len_kmer:
                count += 1
                i = i - j + 1
                j = 0
        else:
            i = i - j + 1
            j = 0
    return count


def feature_extraction(args):
    if args.type == 'DNA':
        alphabet = "ATCG"
    elif args.type == 'RNA':
        alphabet = "AUCG"
    elif args.type == "Protein":
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
    # 准备阶段
    # print("=================Feature extraction step=================")
    input_file_combined = combine_seq_file(args.seq_file, args.result_dir)
    num_list, len_list = count_num(args.type, args.seq_file, args.label, input_file_combined)
    label_list = generate_label_list(num_list, args.label)
    args.fixed_len = max(len_list)

    # 参数生成
    possible_parameter_dict = {}
    possible_parameter_dict = possible_parameter_generation(args, possible_parameter_dict)
    possible_parameter_list = dict_to_list(possible_parameter_dict)

    # 训练/测试集划分
    args = divide_data_set(args, label_list)

    print('Input file direction: '+input_file_combined)
    print('Num of sequence: '+str(len(len_list)))
    print('Num of positive sequence: '+str(num_list[0]))
    print('Num of negative sequence: '+str(num_list[1]))

    output_file = 'input_file_encoded.txt'
    output_list = []
    output_array = []
    if args.code == 'One-hot':
        with open(input_file_combined, 'r') as f:
            for line in f.readlines():
                if line[0] == '>':
                    continue
                temp_line = encode_line_onehot(args, line)
                output_list.append(temp_line)
        width = len(output_list[0][0])
        for i in range(len(output_list)):
            temp_array = np.zeros((args.fixed_len, width))
            temp_len = len(output_list[i])
            if temp_len <= args.fixed_len:
                temp_array[:temp_len, :] = output_list[i]
            output_array.append(temp_array.flatten().tolist())
        output_array = np.array(output_array)
    elif args.code == 'BOW':
        kmer_list = generate_kmer_list(args.word_size, alphabet)
        with open(input_file_combined, 'r') as f:
            for line in f.readlines():
                if line[0] == '>':
                    continue
                sum = 0
                kmer_count_dict = {}
                for kmer in kmer_list:
                    count_temp = kmer_frequency_count(kmer, line)
                    if kmer not in kmer_count_dict:
                        kmer_count_dict[kmer] = 0
                    kmer_count_dict[kmer] += count_temp

                    sum += count_temp

                kmer_count_list = [kmer_count_dict[kmer] for kmer in kmer_list]
                kmer_count = [round(float(kmer) / sum, 8) for kmer in kmer_count_list]
                output_list.append(kmer_count)
        output_array = np.array(output_list)
    elif args.code == 'WE':
        sentences_list = []
        with open(input_file_combined, 'r') as f:
            for line in f.readlines():
                if line[0] == '>':
                    continue
                sent = []
                if len(line) <= args.fixed_len:
                    for j in range(args.fixed_len - len(line)):
                        line += 'X'
                else:
                    line = line[:args.fixed_len]
                for i in range(len(line) - args.word_size[0] + 1):
                    word = line[i:i + args.word_size[0]]
                    sent.append(word)
                sentences_list.append(sent)
        row = (args.fixed_len - args.word_size[0] + 1) * 10
        output = -np.ones((len(sentences_list), row))
        for i, (train, test) in enumerate(args.folds):
            print('Round [%s]' % (i+1))
            train_sentences = []
            test_sentences = []
            for x in train:
                train_sentences.append(sentences_list[x])
            for y in test:
                test_sentences.append(sentences_list[y])
            model = Word2Vec(train_sentences, size=10, window=5, sg=0)
            vectors = []
            for sentence in test_sentences:
                vector = []
                for j in range(len(sentence)):
                    try:
                        temp = np.array(model[sentence[j]])
                    except KeyError:
                        temp = np.zeros(10)

                    if len(vector) == 0:
                        vector = temp
                    else:
                        vector = np.hstack((vector, temp))
                vectors.append(vector)
            for k in range(len(test)):
                output[test[k]] = np.array(vectors[k])
            # output[test] = np.array(vectors)
        output_array = output

    with open(args.result_dir+output_file, 'w') as f:
        for line in output_list:
            f.write(str(line))
            f.write('\n')

    return output_array, label_list, possible_parameter_list








