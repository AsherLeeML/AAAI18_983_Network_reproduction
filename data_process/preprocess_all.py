# Converting EEG to 3D
import os
import pyedflib
import numpy as np
import pandas as pd
import pickle

"""
pre-defined parameters:
    (string)    dataset directory
    (int)       window size
    (bool)      parallizing for paraNet
    (bool)      1D -> 2D converting
    (bool)      segment or not
    (bool)      ignore bad data or not 
"""

dataset_dir = '/home/xdlls/workspace/rep983/extracted_data/'
window_size = 10
ignore_89 = True
output_dir = '/home/xdlls/workspace/rep983/processed/3D_data/'


def data_1Dto2D(data):
    data_2D = np.zeros([10, 11])
    data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0, data[21], data[22], data[23], 	   0,  	     0, 	   0, 	 	 0)
    data_2D[1] = (	  	 0, 	   0,  	   	 0, data[24], data[25], data[26], data[27], data[28], 	   	 0,   	   0, 	 	 0)
    data_2D[2] = (	  	 0, data[29], data[30], data[31], data[32], data[33], data[34], data[35], data[36], data[37], 	 	 0)
    data_2D[3] = (	  	 0, data[38],  data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6], data[39], 		 0)
    data_2D[4] = (data[42], data[40],  data[7],  data[8],  data[9], data[10], data[11], data[12], data[13], data[41], data[43])
    data_2D[5] = (	  	 0, data[44], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[45], 		 0)
    data_2D[6] = (	  	 0, data[46], data[47], data[48], data[49], data[50], data[51], data[52], data[53], data[54], 		 0)
    data_2D[7] = (	  	 0, 	   0, 	 	 0, data[55], data[56], data[57], data[58], data[59], 	   	 0, 	   0, 		 0)
    data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0, data[60], data[61], data[62], 	   0, 	   	 0, 	   0, 		 0)
    data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0, data[63], 		 0, 	   0, 	   	 0, 	   0, 		 0)
    return data_2D


def dataset_convert(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    return dataset_2D


def z_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    # mean_data = np.mean(data[data.nonzero()])
    # sigma_data = np.std(data[data.nonzero()])
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    return data_normalized


def normalize_data(dataset_1D):
    normalized_dataset_1D = np.zeros([dataset_1D.shape[0], 64])
    for i in range(dataset_1D.shape[0]):
        normalized_dataset_1D[i] = z_normalize(dataset_1D[i])
    return normalized_dataset_1D


def windowize(data, size):
    start = 0
    while (start+size) < data.shape[0]:
        yield int(start), int(start+size)
        start += (size/2)


def segment_without_transition(data, label, window_size):
    for (start, end) in windowize(data, window_size):
        if (len(data[start:end]) == window_size) and (len(set(label[start:end]))==1):
            if start == 0:
                segments = data[start:end]
                labels = np.array(list(set(label[start:end])))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(list(set(label[start:end]))))
    return segments, labels


def apply_mixup(dataset_dir, window_size, ignore_89):
    for j in range(1, 110):
        train_label_inter = np.empty([0])
        train_data_inter = np.empty([0, window_size, 10, 11])
        test_label_inter = np.empty([0])
        test_data_inter = np.empty([0, window_size, 10, 11])
        if ignore_89 and j==89:
            continue
        data_dir = dataset_dir + "S" + format(j, "03d")
        task_list = [task for task in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, task))]
        print('Sub'+format(j, '03d') + 'begins:')
        outfile_for_subidx = output_dir + 'sub' + format(j, '03d')+'/'
        if not os.path.isdir(outfile_for_subidx):
            os.system('mkdir '+ outfile_for_subidx)
        for task in task_list:
            if ("R04" in task) or ("R06" in task) or ("R08" in task) or ("R10" in task):
                data_file = data_dir + '/' + task + '/' + task + '.data.csv'
                label_file = data_dir + '/' + task + '/' + task + '.label.csv'
                data  = pd.read_csv(data_file)
                label = pd.read_csv(label_file)
                data_label = pd.concat([data, label], axis=1)
                data_label = data_label.loc[data_label['labels']!='rest']
                label = data_label['labels']
                data_label.drop('labels', axis=1, inplace=True)
                data = data_label.as_matrix()
                data = normalize_data(data)
                data = dataset_convert(data)
                data, label = segment_without_transition(data, label, window_size)
                data = data.reshape(int(data.shape[0]/window_size), window_size, 10, 11)
                train_data_inter = np.vstack([train_data_inter, data])
                train_label_inter = np.append(train_label_inter, label)
            if  ("R12" in task) or ("R14" in task):
                data_file = data_dir + '/' + task + '/' + task + '.data.csv'
                label_file = data_dir + '/' + task + '/' + task + '.label.csv'
                data = pd.read_csv(data_file)
                label = pd.read_csv(label_file)
                data_label = pd.concat([data, label], axis=1)
                data_label = data_label.loc[data_label['labels'] != 'rest']
                label = data_label['labels']
                data_label.drop('labels', axis=1, inplace=True)
                data = data_label.as_matrix()
                data = normalize_data(data)
                data = dataset_convert(data)
                data, label = segment_without_transition(data, label, window_size)
                data = data.reshape(int(data.shape[0] / window_size), window_size, 10, 11)
                test_data_inter = np.vstack([test_data_inter, data])
                test_label_inter = np.append(test_label_inter, label)
        print('end.')
        output_train_data = outfile_for_subidx + "train_data_winsize_" + str(window_size) + ".pkl"
        output_train_label = outfile_for_subidx + "train_label_winsize_" + str(window_size) + '.pkl'
        output_test_data = outfile_for_subidx + 'test_data_winsize_' + str(window_size) + '.pkl'
        output_test_label = outfile_for_subidx + 'test_label_winsize_' + str(window_size) + '.pkl'
        with open(output_train_data, 'wb') as fp:
            pickle.dump(train_data_inter, fp)
        with open(output_train_label, 'wb') as fp:
            pickle.dump(train_label_inter, fp)
        with open(output_test_data, 'wb') as fp:
            pickle.dump(test_data_inter, fp)
        with open(output_test_label, 'wb') as fp:
            pickle.dump(test_label_inter, fp)
    return None




if __name__ == "__main__":
    print("Converting begins:")
    apply_mixup(dataset_dir, window_size, ignore_89)


    # with open(output_data, 'wb') as fp:
    #     pickle.dump(shuffled_data, fp, protocal=4)
    # with open(output_label, 'wb') as fp:
    #     pickle.dump(shuffled_label, fp)
