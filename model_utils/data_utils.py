import pickle
import numpy as np

window_size = 10
ignore_sub = 89
data_dir = './processed/3D_data/'


def load_data_with_label(subject_index, split=True):
    data_path = data_dir + 'sub' +format(subject_index, '03d') + '/'
    with open(data_path+'train_data_winsize_10.pkl', 'rb') as df:
        train_data = pickle.load(df)
    with open(data_path+'train_label_winsize_10.pkl', 'rb') as lf:
        train_label = pickle.load(lf)
    with open(data_path + 'test_data_winsize_10.pkl', 'rb') as df:
        test_data = pickle.load(df)
    with open(data_path + 'test_label_winsize_10.pkl', 'rb') as df:
        test_label = pickle.load(df)
    if split:
        return train_data, train_label, test_data, test_label
    else:
        train_data = np.vstack([train_data, test_data])
        train_label = np.append(train_label, test_label)

    return train_data, train_label


def leave_one_subject_split(left_subject):
    train_data = np.empty([0, window_size, 10, 11])
    train_label = np.empty([0])
    test_data = np.empty([0, window_size, 10, 11])
    test_label = np.empty([0])
    for i in range(1, 110):
        if i == left_subject:
            train_data_temp, train_label_temp, test_data, test_label = load_data_with_label(left_subject, split=True)
            test_data = np.vstack([test_data, train_data_temp])
            test_label = np.append(test_label, train_label_temp)
        elif i == ignore_sub:
            continue
        else:
            data_t1, label_t1, data_t2, label_t2= load_data_with_label(i, split=False)
            data = np.vstack([data_t1, data_t2])
            label = np.append(label_t1, label_t2)
            train_data = np.vstack([train_data, data])
            train_label = np.append(train_label, label)
    return train_data, train_label, test_data, test_label


def leave_one_finetune_split(left_subject, split=False):
    pre_train_data = np.empty([0, window_size, 10, 11])
    pre_train_label = np.empty([0])
    finetune_train_data = np.empty([0, window_size, 10, 11])
    finetune_train_label = np.empty([0])
    test_data = np.empty([0, window_size, 10, 11])
    test_label = np.empty([0])
    for i in range(1, 110):
        if i == left_subject:
            finetune_train_data, finetune_train_label, test_data, test_label = load_data_with_label(left_subject, split=True)
        elif np.random.random() < 0.4 or i == ignore_sub:
            continue
        else:
            data, label = load_data_with_label(i, split=False)
            pre_train_data = np.vstack([pre_train_data, data])
            pre_train_label = np.append(pre_train_label, label)
    if split:
        return pre_train_data, pre_train_label, finetune_train_data, finetune_train_label, test_data, test_label
    else:
        train_data = np.vstack([pre_train_data, finetune_train_data])
        train_label = np.append(pre_train_label, finetune_train_label)
        return train_data, train_label, test_data, test_label
