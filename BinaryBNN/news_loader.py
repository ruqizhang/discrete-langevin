import pandas as pd
from sklearn.model_selection import train_test_split  
import numpy as np

def generate_one_hot_mat(mat):
    upper_bound = np.max(mat)
    mat_one_hot = np.zeros((mat.shape[0], int(upper_bound+1)))
    
    for j in range(mat.shape[0]):
        mat_one_hot[j, int(mat[j])] = 1.
        
    return mat_one_hot



def normalize_data_ours(data_train, data_test):
    ### in this function, we normalize all the data to [0, 1], and bring education_num, capital gain, hours per week to the first three columns, norm to [0, 1]
    n_train = data_train.shape[0]
    n_test = data_test.shape[0]
    data_feature = np.concatenate((data_train, data_test), axis=0)
    
    data_feature_normalized = np.zeros((n_train+n_test, 1))
    class_list = []
    mono_list = []
#     class_list = [1, 3, 5, 6, 7, 8, 9, 13]
#     mono_list = [4, 10, 12]
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


def generate_normalize_numerical_mat(mat):
    mat = (mat - np.min(mat))/(np.max(mat) - np.min(mat))
    return mat

def load_data(get_categorical_info=False):
    news_data = pd.read_csv('./data/news/OnlineNewsPopularity/OnlineNewsPopularity.csv')
    news_data = news_data.drop('url', axis=1)
    news_data = news_data.drop(' timedelta', axis=1)
    #for col in news_data.columns:
    #    print(col)
    news_data = news_data.values
    #print(news_data.shape)

    X, y = news_data[:, :-1], news_data[:, -1]
    #print(X.shape, y.shape)
    #assert False
    y = np.log(y)
    y = (y-y.min())/(y.max() - y.min())
    #print(y.min(), y.max())
    #assert False

    # Normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #print(X_train.shape)
    X_train, X_test, start_index, cat_length = normalize_data_ours(X_train, X_test)

    #print(X_train.shape, X_train[:, 0].min(), X_train[:, 0].max())
    return X_train[:, 1:], y_train, X_test[:, 1:], y_test

if __name__ == '__main__':
    load_data()
