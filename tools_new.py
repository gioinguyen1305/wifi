import numpy as np


def normalize_data1(data):
    m = np.mean(data)
    s = np.std(data)#variance
    new_data = (data-m)/s
    return new_data,m,s


def normalize_data2(data,m,s):
    new_data = (data-m)/s
    return new_data

def normalize_data4(data): #[-1,1]
    m_ax = np.max(data)
    m_in = np.min(data)
    new_data = 2*(data-m_in)/(m_ax - m_in) - 1

    return new_data,m_ax,m_in

def normalize_data5(data,m_ax,m_in): #[-1,1]
    new_data = 2*(data-m_in)/(m_ax - m_in) - 1
    return new_data

def next_batch(data, label, index_set):

    batch_data = data[index_set]
    batch_label = label[index_set]

    return batch_data, batch_label


def next_batch1(data, label, index_set, matrix, location_sample_num):

    SEQUENCE_LEN = 5
    AP_NUM = 5

    l = index_set.shape[0]
    batch_data = np.zeros([l, SEQUENCE_LEN, AP_NUM])
    batch_label = np.zeros([l, 2])

    whole_index = np.zeros([l, SEQUENCE_LEN, 1])

    for i in range(0, l):
        index = int(index_set[i] / location_sample_num)
        pos = matrix[index, 0]
        batch_label[i, :] = label[index_set[i], :]

        if i < l / 2:
            for j in range(0, SEQUENCE_LEN):
                if j != 2:
                    pos_index_set = matrix[index, 2:2 + int(pos)]
                    np.random.shuffle(pos_index_set)
                    region_index_range = np.arange(pos_index_set[0] * location_sample_num, (pos_index_set[0] + 1) * location_sample_num)
                    np.random.shuffle(region_index_range)
                    batch_data[i, j, :] = data[int(region_index_range[0]), :]
                    whole_index[i, j, :] = int(region_index_range[0])
                else:
                    batch_data[i, j, :] = data[index_set[i], :]
                    whole_index[i, j, :] = index_set[i]

        if i >= l / 2:
            index_range = np.arange(index * location_sample_num, (index + 1) * location_sample_num, dtype='i')
            np.random.shuffle(index_range)
            batch_data[i, :, :] = data[index_range[0:5], :]
            for k in range(0, SEQUENCE_LEN):
                whole_index[i, k, :] = index_range[k]

    return batch_data, batch_label


def next_batch2(data, label, index_set, matrix, location_sample_num):

    SEQUENCE_LEN = 5
    AP_NUM = 5

    l = index_set.shape[0]
    batch_data = np.zeros([l, SEQUENCE_LEN, AP_NUM])
    batch_label = np.zeros([l, 2])

    for i in range(0, l):
        index = int(index_set[i] / location_sample_num)
        pos = matrix[index, 0]
        batch_label[i, :] = label[index_set[i], :]

        if i < l / 2:
            for j in range(0, SEQUENCE_LEN):
                if j != 2:
                    pos_index_set = matrix[index, 2:2 + int(pos)]
                    np.random.shuffle(pos_index_set)
                    region_index_range = np.arange(pos_index_set[0] * location_sample_num, (pos_index_set[0] + 1) * location_sample_num)
                    np.random.shuffle(region_index_range)
                    batch_data[i, j, :] = data[int(region_index_range[0]), :]
                else:
                    batch_data[i, j, :] = data[index_set[i], :]

        if i >= l / 2:
            index_range = np.arange(index * location_sample_num, (index + 1) * location_sample_num, dtype='i')
            np.random.shuffle(index_range)
            batch_data[i, 0, :] = data[index_range[0], :]
            batch_data[i, 1, :] = data[index_range[1], :]
            batch_data[i, 2, :] = data[index_range[2], :]
            batch_data[i, 3, :] = data[index_range[3], :]
            batch_data[i, 4, :] = data[index_range[0], :]

    return batch_data, batch_label


def next_batch3(data1, data2, label, index_set, matrix, location_sample_num):

    SEQUENCE_LEN = 5
    AP_NUM = 8

    l = index_set.shape[0]
    batch_data = np.zeros([l, SEQUENCE_LEN, AP_NUM])
    batch_label = np.zeros([l, 2])

    whole_index = np.zeros([l, SEQUENCE_LEN, 1])

    for i in range(0, l):
        index = int(index_set[i] / location_sample_num)
        pos = matrix[index, 0]
        batch_label[i, :] = label[index_set[i], :]

        if i < l / 2:
            if i < l / 4:
                for j in range(0, SEQUENCE_LEN):
                    if j != 2:
                        pos_index_set = matrix[index, 2:2 + pos]
                        np.random.shuffle(pos_index_set)
                        region_index_range = np.arange(pos_index_set[0] * location_sample_num, (pos_index_set[0] + 1) * location_sample_num)
                        np.random.shuffle(region_index_range)
                        batch_data[i, j, :] = data1[int(region_index_range[0]), :]
                        whole_index[i, j, :] = int(region_index_range[0])
                    else:
                        batch_data[i, j, :] = data1[index_set[i], :]
                        whole_index[i, j, :] = index_set[i]
            if i >= l / 4:
                index_range = np.arange(index * location_sample_num, (index + 1) * location_sample_num, dtype='i')
                np.random.shuffle(index_range)
                batch_data[i, :, :] = data1[index_range[0:5], :]
                for k in range(0, SEQUENCE_LEN):
                    whole_index[i, k, :] = index_range[k]

        if i >= l / 2:
            if i < l * 0.75:
                for j in range(0, SEQUENCE_LEN):
                    if j != 2:
                        pos_index_set = matrix[index, 2:2 + pos]
                        np.random.shuffle(pos_index_set)
                        region_index_range = np.arange(pos_index_set[0] * location_sample_num, (pos_index_set[0] + 1) * location_sample_num)
                        np.random.shuffle(region_index_range)
                        batch_data[i, j, :] = data2[int(region_index_range[0]), :]
                        whole_index[i, j, :] = int(region_index_range[0])
                    else:
                        batch_data[i, j, :] = data2[index_set[i], :]
                        whole_index[i, j, :] = index_set[i]
            if i >= l * 0.75:
                index_range = np.arange(index * location_sample_num, (index + 1) * location_sample_num, dtype='i')
                np.random.shuffle(index_range)
                batch_data[i, :, :] = data2[index_range[0:5], :]
                for k in range(0, SEQUENCE_LEN):
                    whole_index[i, k, :] = index_range[k]

    return batch_data, batch_label, whole_index


def next_batch4(data, label, index_set, matrix, location_sample_num):

    SEQUENCE_LEN = 5
    AP_NUM = 8

    l = index_set.shape[0]
    batch_data = np.zeros([l, SEQUENCE_LEN, AP_NUM])
    batch_label = np.zeros([l, 2])

    whole_index = np.zeros([l, SEQUENCE_LEN, 1])

    for i in range(0, l):
        index = int(index_set[i] / location_sample_num)
        pos = matrix[index, 0]
        batch_label[i, :] = label[index_set[i], :]

        if i < l / 2:
            for j in range(0, SEQUENCE_LEN):
                if j != 2:
                    pos_index_set = matrix[index, 2:2 + pos]
                    np.random.shuffle(pos_index_set)
                    region_index_range = np.arange(pos_index_set[0] * location_sample_num, (pos_index_set[0] + 1) * location_sample_num)
                    np.random.shuffle(region_index_range)
                    batch_data[i, j, :] = data[int(region_index_range[0]), :]
                    whole_index[i, j, :] = int(region_index_range[0])
                else:
                    batch_data[i, j, :] = data[index_set[i], :]
                    whole_index[i, j, :] = index_set[i]

        if i >= l / 2:
            index_range = np.arange(index * location_sample_num, (index + 1) * location_sample_num, dtype='i')
            np.random.shuffle(index_range)
            batch_data[i, :, :] = data[index_range[0:5], :]
            for k in range(0, SEQUENCE_LEN):
                whole_index[i, k, :] = index_range[k]

    return batch_data, batch_label


def create_index(index_set, matrix, location_sample_num):
    l = index_set.shape[0]
    new_index_set = np.zeros([l], dtype='i')
    location_index_set = np.zeros([l], dtype='i')

    for i in range(0, l):
        index = int(index_set[i]/location_sample_num)
        pos = matrix[index, 0]
        neg = matrix[index, 1]

        if i < l / 2:
            pos_index_set = matrix[index, 2:2+pos]
            np.random.shuffle(pos_index_set)
            new_index_range = np.arange(pos_index_set[0] * location_sample_num, (pos_index_set[0]+1) * location_sample_num)
            location_index_set[i] = pos_index_set[0]
            np.random.shuffle(new_index_range)
            new_index_set[i] = int(new_index_range[0])

        if i >= l / 2:
            neg_index_set = matrix[index, matrix.shape[1]-neg:matrix.shape[1]]
            np.random.shuffle(neg_index_set)
            new_index_range = np.arange(neg_index_set[0] * location_sample_num, (neg_index_set[0] + 1) * location_sample_num)
            location_index_set[i] = neg_index_set[0]
            np.random.shuffle(new_index_range)
            new_index_set[i] = int(new_index_range[0])

    return new_index_set


def create_index_1(index_set, matrix, location_sample_num):
    l = index_set.shape[0]
    new_index_set = np.zeros([l], dtype='i')
    location_index_set = np.zeros([l], dtype='i')

    for i in range(0, l):
        index = int(index_set[i]/location_sample_num)
        pos = matrix[index, 0]
        neg = matrix[index, 1]

        if i < l / 4:
            pos_index_set = matrix[index, 2:2+pos]
            np.random.shuffle(pos_index_set)
            new_index_range = np.arange(pos_index_set[0] * location_sample_num, (pos_index_set[0]+1) * location_sample_num)
            location_index_set[i] = pos_index_set[0]
            np.random.shuffle(new_index_range)
            new_index_set[i] = int(new_index_range[0])

        if i >= l / 4 and i < l / 2:
            neg_index_set = matrix[index, matrix.shape[1]-neg:matrix.shape[1]]
            np.random.shuffle(neg_index_set)
            new_index_range = np.arange(neg_index_set[0] * location_sample_num, (neg_index_set[0] + 1) * location_sample_num)
            location_index_set[i] = neg_index_set[0]
            np.random.shuffle(new_index_range)
            new_index_set[i] = int(new_index_range[0])

        if i >= l / 2 and i < l * 0.75:
            pos_index_set = matrix[index, 2:2+pos]
            np.random.shuffle(pos_index_set)
            new_index_range = np.arange(pos_index_set[0] * location_sample_num, (pos_index_set[0]+1) * location_sample_num)
            location_index_set[i] = pos_index_set[0]
            np.random.shuffle(new_index_range)
            new_index_set[i] = int(new_index_range[0])

        if i >= l * 0.75 and i < l:
            neg_index_set = matrix[index, matrix.shape[1]-neg:matrix.shape[1]]
            np.random.shuffle(neg_index_set)
            new_index_range = np.arange(neg_index_set[0] * location_sample_num, (neg_index_set[0] + 1) * location_sample_num)
            location_index_set[i] = neg_index_set[0]
            np.random.shuffle(new_index_range)
            new_index_set[i] = int(new_index_range[0])

    return new_index_set


def count_distance(data, point_num):

    sample_num = int(data.shape[0]/point_num)
    distance_matrix = np.zeros([int(point_num*point_num),sample_num,sample_num])
    codistance_matrix = np.zeros([point_num,point_num])
    a = np.zeros([point_num,point_num])
    l = 0

    for i in range(0, point_num):
        for j in range(0, point_num):
            left = data[i * sample_num:(i + 1) * sample_num, :]
            right = data[j * sample_num:(j + 1) * sample_num, :]
            for k in range(0,sample_num):
                distance_matrix[l,:,k] = np.sqrt(np.sum(np.square(left - right[k,:]), 1))

            codistance_matrix[i,j] = np.mean(distance_matrix[l,:,:])
            l = l+1
        diag = codistance_matrix[i,i]
        a[i, :] = codistance_matrix[i, :] - diag
        a[:, i] = codistance_matrix[:, i] - diag

    return codistance_matrix, np.abs(a)

