from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
import os
import math


def plot_pr_curve(cv_labels, cv_prob, file_path):
    precisions = []
    auc_list = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)
    for i in range(len(cv_labels)):
        precision, recall, _ = precision_recall_curve(cv_labels[i], cv_prob[i])
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        try:
            roc_auc = auc(recall, precision)
        except ZeroDivisionError:
            roc_auc = 0.0
        auc_list.append(roc_auc)

    plt.figure(0)
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(auc_list)
    plt.plot(mean_recall, mean_precision, color='navy',
             label=r'Mean PRC (AUPRC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.7)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.3,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.title('Precision-Recall Curve', fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(loc="lower left")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    figure_name = file_path + 'cv_prc.png'
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The Precision-Recall Curve of cross-validation can be found:')
        print(full_path)
        print('\n')
    return mean_auc


def plot_roc_curve(cv_labels, cv_prob, file_path):
    """Plot ROC curve."""
    # Receiver Operating Characteristic
    tpr_list = []
    auc_list = []
    fpr_array = []
    tpr_array = []
    thresholds_array = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(cv_labels)):
        fpr, tpr, thresholds = roc_curve(cv_labels[i], cv_prob[i])
        fpr_array.append(fpr)
        tpr_array.append(tpr)
        thresholds_array.append(thresholds)
        tpr_list.append(interp(mean_fpr, fpr, tpr))
        tpr_list[-1][0] = 0.0
        try:
            roc_auc = auc(fpr, tpr)
        except ZeroDivisionError:
            roc_auc = 0.0
        auc_list.append(roc_auc)

    plt.figure(0)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.7)
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)
    plt.plot(mean_fpr, mean_tpr, color='navy',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.7)
    std_tpr = np.std(tpr_list, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=.3,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.title('Receiver Operating Characteristic', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    figure_name = file_path + 'cv_roc.png'
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The Receiver Operating Characteristic of cross-validation can be found:')
        print(full_path)
        print('\n')
    return mean_auc


def evaluation(label, label_predict):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for i in range(len(label)):
        if label[i] == 1 and label_predict[i] == 1:
            tp += 1.0
        elif label[i] == 1 and label_predict[i] == -1:
            fn += 1.0
        elif label[i] == -1 and label_predict[i] == 1:
            fp += 1.0
        elif label[i] == -1 and label_predict[i] == -1:
            tn += 1.0
    try:
        acc = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        acc = 0.0
    try:
        auc = roc_auc_score(label, label_predict)
    except ZeroDivisionError:
        auc = 0.0
    try:
        sensitivity = tp / (tp + fn)
    except ZeroDivisionError:
        sensitivity = 0.0
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0.0
    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0.0
    try:
        f1 = (2 * p * r) / (p + r)
    except ZeroDivisionError:
        f1 = 0.0
    return acc, auc, sensitivity, specificity, f1


def result_print(result):
    result_dict = {'Accuracy': result[0], 'AUC': result[1], 'Sensitivity': result[2],
                   'Specificity': result[3], 'F1-score': result[4]}
    print("Result: ")
    for i in range(len(result_dict)):
        print(str(list(result_dict.keys())[i]) + ':' + str(list(result_dict.values())[i]))


def performance_evaluation(args, output_array, folds, label_list, best_parameter_pair):
    if args.method == 'SVM':
        temp_str = 'The best parameter for SVM is: cost = ' + str(best_parameter_pair['cost']) + ', gamma = ' + str(best_parameter_pair['gamma'])
        # print(temp_str.center(40, '+'))
        results = []
        true_labels = []
        predict_labels = []
        predict_probability = []
        for train, test in folds:
            x_train = output_array[train]
            x_test = output_array[test]
            y_train = label_list[train]
            y_test = label_list[test]
            classification = svm.SVC(C=2 ** best_parameter_pair['cost'], gamma=2 ** best_parameter_pair['gamma'], probability=True)
            classification.fit(x_train, y_train)
            y_test_predict = classification.predict(x_test)
            y_test_prob_predict = classification.predict_proba(x_test)[:, 1]
            result = evaluation(y_test, y_test_predict)
            results.append(result)
            true_labels.append(y_test)
            predict_labels.append(y_test_predict)
            predict_probability.append(y_test_prob_predict)
        plot_roc_curve(true_labels, predict_probability, args.result_dir)
        plot_pr_curve(true_labels, predict_probability, args.result_dir)
        final_result = np.array(results).mean(axis=0)
        result_print(final_result)

    elif args.method == 'LinearSVM':
        temp_str = 'The best parameter for Linear SVM is: cost = ' + str(best_parameter_pair['cost'])
        # print(temp_str.center(40, '+'))
        results = []
        true_labels = []
        predict_labels = []
        predict_probability = []
        for train, test in folds:
            x_train = output_array[train]
            x_test = output_array[test]
            y_train = label_list[train]
            y_test = label_list[test]
            classification = svm.SVC(C=2 ** best_parameter_pair['cost'], kernel="linear", probability=True)
            classification.fit(x_train, y_train)
            y_test_predict = classification.predict(x_test)
            y_test_prob_predict = classification.predict_proba(x_test)[:, 1]
            result = evaluation(y_test, y_test_predict)
            results.append(result)
            true_labels.append(y_test)
            predict_labels.append(y_test_predict)
            predict_probability.append(y_test_prob_predict)
        plot_roc_curve(true_labels, predict_probability, args.result_dir)
        plot_pr_curve(true_labels, predict_probability, args.result_dir)
        final_result = np.array(results).mean(axis=0)
        result_print(final_result)

    elif args.method == 'RF':
        temp_str = 'The best parameter for RF is: tree = ' + str(best_parameter_pair['tree'])
        # print(temp_str.center(40, '+'))
        results = []
        true_labels = []
        predict_labels = []
        predict_probability = []
        for train, test in folds:
            x_train = output_array[train]
            x_test = output_array[test]
            y_train = label_list[train]
            y_test = label_list[test]
            classification = RandomForestClassifier(random_state=42, n_estimators=best_parameter_pair['tree'])
            classification.fit(x_train, y_train)
            y_test_predict = classification.predict(x_test)
            y_test_prob_predict = classification.predict_proba(x_test)[:, 1]
            result = evaluation(y_test, y_test_predict)
            results.append(result)
            true_labels.append(y_test)
            predict_labels.append(y_test_predict)
            predict_probability.append(y_test_prob_predict)
        plot_roc_curve(true_labels, predict_probability, args.result_dir)
        plot_pr_curve(true_labels, predict_probability, args.result_dir)
        final_result = np.array(results).mean(axis=0)
        result_print(final_result)

    elif args.method == 'KNN':
        temp_str = 'The best parameter for KNN is: neighbors = ' + str(best_parameter_pair['ngb'])
        # print(temp_str.center(40, '+'))
        results = []
        true_labels = []
        predict_labels = []
        predict_probability = []
        for train, test in folds:
            x_train = output_array[train]
            x_test = output_array[test]
            y_train = label_list[train]
            y_test = label_list[test]
            classification = KNeighborsClassifier(n_neighbors=best_parameter_pair['ngb'])
            classification.fit(x_train, y_train)
            y_test_predict = classification.predict(x_test)
            y_test_prob_predict = classification.predict_proba(x_test)[:, 1]
            result = evaluation(y_test, y_test_predict)
            results.append(result)
            true_labels.append(y_test)
            predict_labels.append(y_test_predict)
            predict_probability.append(y_test_prob_predict)
        plot_roc_curve(true_labels, predict_probability, args.result_dir)
        plot_pr_curve(true_labels, predict_probability, args.result_dir)
        final_result = np.array(results).mean(axis=0)
        result_print(final_result)

    elif args.method == 'AdaBoost' or args.method == 'NB' or args.method == 'LDA' or args.method == 'QDA':
        results = []
        true_labels = []
        predict_labels = []
        predict_probability = []
        for train, test in folds:
            x_train = output_array[train]
            x_test = output_array[test]
            y_train = label_list[train]
            y_test = label_list[test]
            if args.method == 'AdaBoost':
                classification = AdaBoostClassifier()
            elif args.method == 'NB':
                classification = GaussianNB()
            elif args.method == 'LDA':
                classification = lda()
            elif args.method == 'QDA':
                classification = qda()
            classification.fit(x_train, y_train)
            y_test_predict = classification.predict(x_test)
            y_test_prob_predict = classification.predict_proba(x_test)[:, 1]
            result = evaluation(y_test, y_test_predict)
            results.append(result)
            true_labels.append(y_test)
            predict_labels.append(y_test_predict)
            predict_probability.append(y_test_prob_predict)
        plot_roc_curve(true_labels, predict_probability, args.result_dir)
        plot_pr_curve(true_labels, predict_probability, args.result_dir)
        final_result = np.array(results).mean(axis=0)
        result_print(final_result)

    all_predict = classification.predict(output_array)
    with open(args.result_dir + 'prediction result', 'w') as f:
        space = '          '
        f.write('No.' + space + 'True Label' + space + 'Predict Label\n')
        for i in range(len(all_predict)):
            f.write(str(i) + space + str(label_list[i]) + space + str(all_predict[i]))
            f.write('\n')



