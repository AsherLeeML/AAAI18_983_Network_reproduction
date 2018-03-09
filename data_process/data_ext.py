import os
import numpy as np
import pandas as pd
import pyedflib

dataset_path = '/home/xdlls/workspace/dataset/eegmmidb/'
target_path  = '/home/xdlls/workspace/rep983/extracted_data/'


def read_data(file_name):
    with pyedflib.EdfReader(file_name) as f:
        n = f.signals_in_file
        signal_channels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)
        sigbufs = np.transpose(sigbufs)
        signal = np.asarray(sigbufs)
        signal_channels = np.asarray(signal_channels)
        return signal, signal_channels


def gen_label(target_path):
    sub_list = os.listdir(target_path)
    for j in range(len(sub_list)):
        if os.path.isdir(target_path + sub_list[j]):
            sub_file = target_path+sub_list[j].strip()
            os.chdir(sub_file)
            sub_data_list = os.listdir()
            for k in range(len(sub_data_list)):
                file_name = sub_data_list[k]
                print(file_name + " begins:")
                eegdata = pd.read_csv(file_name+"/"+file_name+'.data.csv')
                label_NBs = eegdata.shape[0]
                if "R01" in file_name:
                    labels = pd.DataFrame(['eye_open'] * label_NBs, columns=['labels'])
                    labels.to_csv(file_name+'/'+file_name+'.label.csv', index=False)
                elif "R02" in file_name:
                    labels = pd.DataFrame(['eye_close'] * label_NBs, columns=['labels'])
                    labels.to_csv(file_name+'/'+file_name+'.label.csv', index=False)
                elif "R03" in file_name or "R07" in file_name or "R11" in file_name:
                    event = pd.read_fwf(file_name+'/'+file_name+'.event.csv')
                    event['Aux'] = event['Aux'].astype(str)
                    labels = []
                    for p in range(len(event['Aux'])):
                        if p == (len(event['Aux'])-1):
                            label_number = label_NBs - event['Sample #'][p]
                        else:
                            label_number = event['Sample #'][p+1] - event['Sample #'][p]
                        if 'T0' in event['Aux'][p]:
                            labels = labels + ['rest']*label_number
                        elif 'T1' in event['Aux'][p]:
                            labels = labels + ['open&close_left_fist']*label_number
                        elif 'T2' in event['Aux'][p]:
                            labels = labels + ['open&close_right_fist']*label_number
                        else:
                            print(event['Aux'][p])
                    labels = pd.DataFrame(labels, columns=['labels'])
                    labels.to_csv(file_name+'/'+file_name+'.label.csv', index=False)
                elif "R04" in file_name or "R08" in file_name or "R12" in file_name:
                    event = pd.read_fwf(file_name+'/'+file_name+'.event.csv')
                    event['Aux'] = event['Aux'].astype(str)
                    labels = []
                    for p in range(len(event['Aux'])):
                        if p == (len(event['Aux'])-1):
                            label_number = label_NBs - event['Sample #'][p]
                        else:
                            label_number = event['Sample #'][p+1] - event['Sample #'][p]
                        if 'T0' in event['Aux'][p]:
                            labels = labels + ['rest']*label_number
                        elif 'T1' in event['Aux'][p]:
                            labels = labels + ['imagine_open&close_left_fist']*label_number
                        elif 'T2' in event['Aux'][p]:
                            labels = labels + ['imagine_open&close_right_fist']*label_number
                        else:
                            print(event['Aux'][p])
                    labels = pd.DataFrame(labels, columns=['labels'])
                    labels.to_csv(file_name+'/'+file_name+'.label.csv', index=False)
                elif "R05" in file_name or "R09" in file_name or "R13" in file_name:
                    event = pd.read_fwf(file_name+'/'+file_name+'.event.csv')
                    event['Aux'] = event['Aux'].astype(str)
                    labels = []
                    for p in range(len(event['Aux'])):
                        if p == (len(event['Aux']) - 1):
                            label_number = label_NBs - event['Sample #'][p]
                        else:
                            label_number = event['Sample #'][p + 1] - event['Sample #'][p]
                        if 'T0' in event['Aux'][p]:
                            labels = labels + ['rest'] * label_number
                        elif 'T1' in event['Aux'][p]:
                            labels = labels + ['open&close_both_fists'] * label_number
                        elif 'T2' in event['Aux'][p]:
                            labels = labels + ['open&close_both_feet'] * label_number
                        else:
                            print(event['Aux'][p])
                    labels = pd.DataFrame(labels, columns=['labels'])
                    labels.to_csv(file_name+'/'+file_name+'.label.csv', index=False)
                elif "R06" in file_name or "R10" in file_name or "R14" in file_name:
                    event = pd.read_fwf(file_name+'/'+file_name+'.event.csv')
                    event['Aux'] = event['Aux'].astype(str)
                    labels = []
                    for p in range(len(event['Aux'])):
                        if p == (len(event['Aux']) - 1):
                            label_number = label_NBs - event['Sample #'][p]
                        else:
                            label_number = event['Sample #'][p + 1] - event['Sample #'][p]
                        if 'T0' in event['Aux'][p]:
                            labels = labels + ['rest'] * label_number
                        elif 'T1' in event['Aux'][p]:
                            labels = labels + ['imagine_open&close_both_fists'] * label_number
                        elif 'T2' in event['Aux'][p]:
                            labels = labels + ['imagine_open&close_both_feet'] * label_number
                        else:
                            print(event['Aux'][p])
                    labels = pd.DataFrame(labels, columns=['labels'])
                    labels.to_csv(file_name+'/'+file_name+'.label.csv', index=False)
                else:
                    print("***********ERROR***************")
                if len(labels['labels']) == label_NBs:
                    pass
                    #print("OK.")
                else:
                    print(len(labels['labels']))
                    print(label_NBs)
                    print("ERROR")
                print(file_name + 'END\n')
        else:
            pass
    return None


def convert_edf_to_csv(dataset_path, target_path):
    sub_list = os.listdir(dataset_path)
    for j in range(len(sub_list)):
        if os.path.isdir(dataset_path + sub_list[j]):
            print("*************************************")
            print("Subject NO."+sub_list[j].strip('S')+" Converting:")
            sub_name = sub_list[j].strip()
            sub_file = dataset_path + sub_name
            os.chdir(sub_file)
            sub_data_list = os.popen("ls *.edf").readlines()

            for k in range(len(sub_data_list)):
                file_name = os.path.split(sub_data_list[k])[1].strip()
                print(file_name + " begins:")
                gen_file_name = file_name.rsplit('.edf')[0]
                # eeg data extracting
                eeg_data_1d, channels = read_data(file_name)
                eeg_data_1d = pd.DataFrame(eeg_data_1d, columns=[list(channels)])
                csv_data_file = target_path + sub_name + '/' + gen_file_name
                if not os.path.exists(csv_data_file):
                    os.system("mkdir -p " + csv_data_file)
                gen_name = csv_data_file + '/' + gen_file_name
                eeg_data_1d.to_csv(gen_name+'.data.csv', index=False)
                # *.event.csv generating
                os.system("rdann -r " + file_name + " -f 0 -t 125 -a event -v >" + gen_name + ".event.csv")
    return None


if __name__ == "__main__":
    convert_edf_to_csv(dataset_path, target_path)
    gen_label(target_path)
