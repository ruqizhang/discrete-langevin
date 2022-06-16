import numpy as np
import pandas as pd

def generate_one_hot_mat(mat):
    upper_bound = np.max(mat)
    mat_one_hot = np.zeros((mat.shape[0], int(upper_bound+1)))
    
    for j in range(mat.shape[0]):
        mat_one_hot[j, int(mat[j])] = 1.
        
    return mat_one_hot

def generate_normalize_numerical_mat(mat):
    if np.max(mat) == np.min(mat):
        return mat
    mat = (mat - np.min(mat))/(np.max(mat) - np.min(mat))
    #mat = 2 * (mat - 0.5)
    
    return mat
    
def normalize_data_ours(data_train, data_test):
    ### in this function, we normalize all the data to [0, 1], and bring education_num, capital gain, hours per week to the first three columns, norm to [0, 1]
    n_train = data_train.shape[0]
    n_test = data_test.shape[0]
    data_feature = np.concatenate((data_train, data_test), axis=0)
    
    data_feature_normalized = np.zeros((n_train+n_test, 1))
    class_list = []
    mono_list = [50, 51, 52, 53, 55, 56, 57, 58]
    ### store the class variables
    start_index = []
    cat_length = []
    ### Normalize Mono Features
    for i in range(data_feature.shape[1]):
        if i in mono_list:
            if i == mono_list[0]:
                mat = data_feature[:, i]
                mat = mat[:, np.newaxis]
                data_feature_normalized = generate_normalize_numerical_mat(mat)
            else:
                mat = data_feature[:, i]
                mat = generate_normalize_numerical_mat(mat)
                mat = mat[:, np.newaxis]
                #print(adult_feature_normalized.shape, mat.shape)
                data_feature_normalized = np.concatenate((data_feature_normalized, mat), axis=1)
        else:
            continue
    ### Normalize non-mono features and turn class labels to one-hot vectors
    for i in range(data_feature.shape[1]):
        if i in mono_list:
            continue
        elif i in class_list:
            continue
        else:
            mat = data_feature[:, i]
            if np.max(mat) == np.min(mat):
                continue
            mat = generate_normalize_numerical_mat(mat)
            mat = mat[:, np.newaxis]
            data_feature_normalized = np.concatenate((data_feature_normalized, mat), axis=1)
    
    for i in range(data_feature.shape[1]):
        if i in mono_list:
            continue
        elif i in class_list:
            mat = data_feature[:, i]
            mat = generate_one_hot_mat(mat)
            start_index.append(data_feature_normalized.shape[1])
            cat_length.append(mat.shape[1])
            data_feature_normalized = np.concatenate((data_feature_normalized, mat), axis=1)
        else:
            continue
    
    data_train = data_feature_normalized[:n_train, :]
    data_test = data_feature_normalized[n_train:, :]
    
    assert data_test.shape[0] == n_test
    assert data_train.shape[0] == n_train
    
    return data_train, data_test, start_index, cat_length 

def load_data(get_categorical_info=True):

    data_train = pd.read_csv('./data/blog/train.csv')
    data_test = pd.read_csv('./data/blog/test.csv')

    data_train = np.array(data_train.values)
    data_test = np.array(data_test.values)
    
    
    X_train = data_train[:, :280].astype(np.float64)
    y_train = data_train[:, 280].astype(np.uint8)

    X_test = data_test[:, :280].astype(np.float64)
    y_test = data_test[:, 280].astype(np.uint8)
    
    y = np.concatenate([y_train, y_test], axis=0)
    q = np.percentile(y, 90)
    #print(q)
    cols = []
    for i in range(y_train.shape[0]):
        if y_train[i] > q:
            cols.append(i)
    X_train=np.delete(X_train, cols, axis=0)
    y_train=np.delete(y_train, cols, axis=0)
    
    cols = []
    for i in range(y_test.shape[0]):
        if y_test[i] > q:
            cols.append(i)
    X_test=np.delete(X_test, cols, axis=0)
    y_test=np.delete(y_test, cols, axis=0)
    X_train, X_test, start_index, cat_length = normalize_data_ours(X_train, X_test)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 
    #mono_list = [50, 51, 52, 53, 55, 56, 57, 58]
    #import matplotlib.pyplot as plt
    #plt.scatter(X_train[:, 4], y_train)
    #plt.scatter(X_train[:, 2], y_train)
    #fig = plt.gcf()
    #fig.savefig('dd.png')
    normalized_y = generate_normalize_numerical_mat(np.concatenate([y_train, y_test], axis=0))
    y_train = normalized_y[:y_train.shape[0]]
    y_test = normalized_y[y_train.shape[0]:]

    #ll = 0
    #for i in range(y_train.shape[0]):
    #    if y_train[i]>0:
    #        ll += 1
    #print(ll)
    #print(np.max(y_train), np.min(y_train), np.median(y_train))
    
    if get_categorical_info:
        return X_train, y_train, X_test, y_test, start_index, cat_length 
    else:
        return X_train, y_train, X_test, y_test

load_data()
