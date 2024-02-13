import numpy as np
import csv
import os


def load_data(data, seq_len=15, his=1, pre_sens_num=1):
    max = np.max(data)
    min = np.min(data)
    med = max - min
    data = np.array(data, dtype=float)
    data_nor = (data - min) / med

    sequence_length = seq_len + his
    result = []
    for index in range(len(data_nor) - sequence_length):
        result.append(data_nor[index: index + sequence_length])
    result = np.stack(result, axis=0)
    train = result[:]
    x_train = train[:, :seq_len]
    x_wd_train = train[:, :seq_len, pre_sens_num-1]
    y_train = train[:, -1, pre_sens_num-1]
    x_data = []
    x_w = []
    x_d = []
    label = []
    for i in range (len(train)):
        if i >= 2016:
            x_data.append(x_train[i])
            x_w.append(x_wd_train[i - 2016 + 8])
            x_d.append(x_wd_train[i - 288 + 8])
            label.append(y_train[i])
    x_data = np.array(x_data)
    x_w = np.array(x_w)
    x_d = np.array(x_d)
    label = np.array(label)
    return x_data,x_w,x_d,label,med,min

def generate_data(data1, data2, seq_len, pre_len, pre_sens_num):
    data = np.stack((data1, data2), axis=1)
    x_data, x_w, x_d, label, med, min = load_data(data, seq_len ,pre_len, pre_sens_num)

    row = 2016
    train_x_data = x_data[:-row]
    test_data = x_data[-row:]
    train_w = x_w[:-row]
    test_w = x_w[-row:]
    train_d = x_d[:-row]
    test_d = x_d[-row:]
    train_l = label[:-row]
    test_l = label[-row:]
    return train_x_data, train_w, train_d, train_l, \
           test_data, test_w, test_d, test_l, med, min



def load_csv(fir_dir, col, scenario):
    file_all = []
    if scenario == "temperature1":
        file_all=['17_temperature.csv','18_temperature.csv','19_temperature.csv',
                  '21_temperature.csv','25_temperature.csv']
    if scenario == "temperature2":
        file_all=['26_temperature.csv','27_temperature.csv','28_temperature.csv',
                  '29_temperature.csv','30_temperature.csv']
    file_name = []
    for i in file_all:
        file_name.append(os.path.join(fir_dir, i))
    all_data = []
    for filename in file_name:
        csvfile = open(filename,'r')
        reader = csv.reader(csvfile)
        a = []
        for line in reader:

            a.append(line)
        b = []
        for i in range(len(a)):
            b.append(a[i][col])
        data = b[1:]
        all_data.extend(data)
        csvfile.close()
    #for i in range(len(all_data)):
    #    print(i, all_data[i])
    #    print(i, all_data[i], float(all_data[i]))
    data = np.array(all_data,dtype=float)
    return data
