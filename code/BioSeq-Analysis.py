import os
import time
import shutil
from FeatureExtraction import feature_extraction
from MachineLearning import machine_learning
from PerformanceEvaluation import performance_evaluation


def main(args):
    print("\nAnalysis Start\n")

    start_time = time.time()
    current_path = os.path.dirname(os.path.realpath(__file__))
    args.current_dir = os.path.dirname(os.getcwd())
    args.result_dir = args.current_dir + '/results/'
    if not os.path.exists(args.result_dir):
        try:
            os.makedirs(args.result_dir)
            print('result_dir: ', args.result_dir)
        except OSError:
            pass
    else:
        try:
            shutil.rmtree(args.result_dir)
            print('result_dir: ', args.result_dir)
            os.makedirs(args.result_dir)
        except OSError:
            pass

    print("=================Prepare step=================")
    print('Sequence type: '+args.type)
    print('Machine learning method: '+args.method)
    print('Result direction: '+args.result_dir)

    print("=================Feature extraction step=================")
    output_array, label_list, possible_parameter_list = feature_extraction(args)

    print("=================Parameter selection step=================")
    best_parameter_pair = machine_learning(args, output_array, args.folds, label_list, possible_parameter_list)

    print("=================Performance evaluation step=================")
    performance_evaluation(args, output_array, args.folds, label_list, best_parameter_pair)

    print("\nAnalysis finished.")
    print("Time = %.2fs" % (time.time()-start_time))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='BioSeq-Analysis')
    # 特征提取
    parse.add_argument('-type', type=str, choices=['DNA', 'RNA', 'Protein'], required=True, help="Type of input sequence.")
    # 编码方式
    parse.add_argument('-code', type=str, choices=['One-hot', 'BOW', 'WE'], required=True)
    parse.add_argument('-word', type=str, choices=['Kmer'])
    parse.add_argument('-word_size', type=int, default=[3])
    # 分类器构建
    parse.add_argument('-method', type=str, choices=['SVM', 'LinearSVM', 'RF', 'KNN', 'AdaBoost', 'NB', 'LDA', 'QDA'], required=True, help="Machine learning method you choose.")
    # 分类器参数——SVM
    parse.add_argument('-cost', type=int, nargs='*')
    parse.add_argument('-gamma', type=int, nargs='*')
    # 分类器参数——RF
    parse.add_argument('-tree', type=int, nargs='*')
    # 分类器参数——KNN
    parse.add_argument('-ngb', type=int, nargs='*')
    # 测试相关参数
    parse.add_argument('-test', choices=['3, 5, 10'], default='5')
    # 输入文件及标签
    parse.add_argument('-seq_file', nargs='*', required=True, help="Input file.")
    parse.add_argument('-fixed_len', type=int)
    parse.add_argument('-label', type=int, nargs='*', required=True, help="The corresponding label of input file.")

    argv = parse.parse_args()
    main(argv)
