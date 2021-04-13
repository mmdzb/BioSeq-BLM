from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def machine_learning(args, output_array, folds, label_list, possible_parameter_list):
    if args.method == 'SVM':
        max_acc = 0
        max_index = 0
        for i in range(len(possible_parameter_list)):
            parameter_pair = possible_parameter_list[i]
            output_str = '  cost = 2 ** ' + str(parameter_pair['cost']) + ' | ' + 'gamma = 2 ** ' + str(parameter_pair['gamma']) + '  '
            print(output_str.center(40, '='))

            cnt = 0
            true_cnt = 0
            for train, test in folds:
                # output_array = np.array(output_array)
                x_train = output_array[train]
                x_test = output_array[test]
                y_train = label_list[train]
                y_test = label_list[test]
                classification = svm.SVC(C=2 ** parameter_pair['cost'], gamma=2 ** parameter_pair['gamma'], probability=True)
                classification.fit(x_train, y_train)
                y_test_predict_prob = classification.predict_proba(x_test)
                y_test_predict = classification.predict(x_test)
                # print("test:")
                # print(y_test)
                # print("predict:")
                # print(y_test_predict)
                for j in range(len(y_test)):
                    cnt += 1
                    if y_test_predict[j] == y_test[j]:
                        true_cnt += 1
            acc = (1.0*true_cnt)/(1.0*cnt)
            print('Acc = ' + str(acc))
            if acc > max_acc:
                max_acc = acc
                max_index = i
        print("best parameter:")
        print('cost:' + str(possible_parameter_list[max_index]['cost']) + ' gamma: ' +
              str(possible_parameter_list[max_index]['gamma']))
        best_parameter_pair = possible_parameter_list[max_index]
        return best_parameter_pair
    elif args.method == 'LinearSVM':
        max_acc = 0
        max_index = 0
        for i in range(len(possible_parameter_list)):
            parameter_pair = possible_parameter_list[i]
            output_str = '  cost = 2 ** ' + str(parameter_pair['cost'])
            print(output_str.center(40, '='))

            cnt = 0
            true_cnt = 0
            for train, test in folds:
                # output_array = np.array(output_array)
                x_train = output_array[train]
                x_test = output_array[test]
                y_train = label_list[train]
                y_test = label_list[test]
                classification = svm.SVC(C=2 ** parameter_pair['cost'], kernel="linear", probability=True)
                classification.fit(x_train, y_train)
                y_test_predict_prob = classification.predict_proba(x_test)
                y_test_predict = classification.predict(x_test)
                # print("test:")
                # print(y_test)
                # print("predict:")
                # print(y_test_predict)
                for j in range(len(y_test)):
                    cnt += 1
                    if y_test_predict[j] == y_test[j]:
                        true_cnt += 1
            acc = (1.0 * true_cnt) / (1.0 * cnt)
            print('Acc = ' + str(acc))
            if acc > max_acc:
                max_acc = acc
                max_index = i
        print("best parameter:")
        print('cost:' + str(possible_parameter_list[max_index]['cost']))
        best_parameter_pair = possible_parameter_list[max_index]
        return best_parameter_pair
    elif args.method == 'RF':
        max_acc = 0
        max_index = 0
        for i in range(len(possible_parameter_list)):
            parameter_pair = possible_parameter_list[i]
            output_str = '  tree = ' + str(parameter_pair['tree'])
            print(output_str.center(40, '='))

            cnt = 0
            true_cnt = 0
            for train, test in folds:
                # output_array = np.array(output_array)
                x_train = output_array[train]
                x_test = output_array[test]
                y_train = label_list[train]
                y_test = label_list[test]
                classification = RandomForestClassifier(random_state=42, n_estimators=parameter_pair['tree'])
                classification.fit(x_train, y_train)
                y_test_predict_prob = classification.predict_proba(x_test)
                y_test_predict = classification.predict(x_test)
                # print("test:")
                # print(y_test)
                # print("predict:")
                # print(y_test_predict)
                for j in range(len(y_test)):
                    cnt += 1
                    if y_test_predict[j] == y_test[j]:
                        true_cnt += 1
            acc = (1.0 * true_cnt) / (1.0 * cnt)
            print('Acc = ' + str(acc))
            if acc > max_acc:
                max_acc = acc
                max_index = i
        print("best parameter:")
        print('tree:' + str(possible_parameter_list[max_index]['tree']))
        best_parameter_pair = possible_parameter_list[max_index]
        return best_parameter_pair
    elif args.method == 'KNN':
        max_acc = 0
        max_index = 0
        for i in range(len(possible_parameter_list)):
            parameter_pair = possible_parameter_list[i]
            output_str = '  neighbors = ' + str(parameter_pair['ngb'])
            print(output_str.center(40, '='))

            cnt = 0
            true_cnt = 0
            for train, test in folds:
                # output_array = np.array(output_array)
                x_train = output_array[train]
                x_test = output_array[test]
                y_train = label_list[train]
                y_test = label_list[test]
                classification = KNeighborsClassifier(n_neighbors=parameter_pair['ngb'])
                classification.fit(x_train, y_train)
                y_test_predict_prob = classification.predict_proba(x_test)
                y_test_predict = classification.predict(x_test)
                # print("test:")
                # print(y_test)
                # print("predict:")
                # print(y_test_predict)
                for j in range(len(y_test)):
                    cnt += 1
                    if y_test_predict[j] == y_test[j]:
                        true_cnt += 1
            acc = (1.0 * true_cnt) / (1.0 * cnt)
            print('Acc = ' + str(acc))
            if acc > max_acc:
                max_acc = acc
                max_index = i
        print("best parameter:")
        print('neighbors:' + str(possible_parameter_list[max_index]['ngb']))
        best_parameter_pair = possible_parameter_list[max_index]
        return best_parameter_pair
    elif args.method == 'AdaBoost' or args.method == 'NB' or args.method == 'LDA' or args.method == 'QDA':
        best_parameter_pair = []
        return best_parameter_pair
