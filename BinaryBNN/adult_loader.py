import pandas as pd
import numpy as np

def generate_one_hot_mat(mat):
    upper_bound = np.max(mat)
    mat_one_hot = np.zeros((mat.shape[0], int(upper_bound+1)))
    
    for j in range(mat.shape[0]):
        mat_one_hot[j, int(mat[j])] = 1.
        
    return mat_one_hot

def generate_normalize_numerical_mat(mat):
    mat = (mat - np.min(mat))/(np.max(mat) - np.min(mat))
    #mat = 2 * (mat - 0.5)
    
    return mat
    
def normalize_data_ours(adult_train, adult_test):
    ### in this function, we normalize all the data to [0, 1], and bring education_num, capital gain, hours per week to the first three columns, norm to [0, 1]
    n_train = adult_train.shape[0]
    n_test = adult_test.shape[0]
    adult_feature = np.concatenate((adult_train, adult_test), axis=0)
    
    adult_feature_normalized = np.zeros((n_train+n_test, 1))
    class_list = [1, 4, 5, 6, 7, 8, 12]
    mono_list = [3, 9, 11]
#     class_list = [1, 3, 5, 6, 7, 8, 9, 13]
#     mono_list = [4, 10, 12]
    ### store the class variables
    start_index = []
    cat_length = []
    ### Normalize Mono Features
    for i in range(adult_feature.shape[1]):
        if i in mono_list:
            if i == mono_list[0]:
                mat = adult_feature[:, i]
                mat = mat[:, np.newaxis]
                adult_feature_normalized = generate_normalize_numerical_mat(mat)
            else:
                mat = adult_feature[:, i]
                mat = generate_normalize_numerical_mat(mat)
                mat = mat[:, np.newaxis]
                #print(adult_feature_normalized.shape, mat.shape)
                adult_feature_normalized = np.concatenate((adult_feature_normalized, mat), axis=1)
        else:
            continue
    ### Normalize non-mono features and turn class labels to one-hot vectors
    for i in range(adult_feature.shape[1]):
        if i in mono_list:
            continue
        elif i in class_list:
            continue
        else:
            mat = adult_feature[:, i]
            mat = generate_normalize_numerical_mat(mat)
            mat = mat[:, np.newaxis]
            adult_feature_normalized = np.concatenate((adult_feature_normalized, mat), axis=1)
    
    for i in range(adult_feature.shape[1]):
        if i in mono_list:
            continue
        elif i in class_list:
            mat = adult_feature[:, i]
            mat = generate_one_hot_mat(mat)
            start_index.append(adult_feature_normalized.shape[1])
            cat_length.append(mat.shape[1])
            adult_feature_normalized = np.concatenate((adult_feature_normalized, mat), axis=1)
        else:
            continue
    
    adult_train = adult_feature_normalized[:n_train, :]
    adult_test = adult_feature_normalized[n_train:, :]
    
    assert adult_test.shape[0] == n_test
    assert adult_train.shape[0] == n_train
    
    return adult_train, adult_test, start_index, cat_length  
    
def load_data(get_categorical_info=False):
    # Add column names to data set
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
           'relationship', 'race','sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    # Read in train data
    adult_train = pd.read_csv('./data/adult.data', header=None, names=columns, skipinitialspace=True)
    adult_test = pd.read_csv('./data/adult.test', header=None, skiprows=1, names=columns, skipinitialspace=True)
    
    adult_train = adult_train.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in adult_train:
        if adult_train[col].dtype == 'object':
            adult_train = adult_train[adult_train[col] != '?']
    
    adult_test = adult_test.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in adult_test:
        if adult_test[col].dtype == 'object':
            adult_test = adult_test[adult_test[col] != '?']

    replace_train = [
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
         'Never-worked'],
        ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th',
         '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
        ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
         'Married-AF-spouse'],
        ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
         'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
         'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
        ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
        ['Female', 'Male'],
        ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
         'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
         'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
         'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
         'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
        ['<=50K', '>50K']
    ]

    for row in replace_train:
        adult_train = adult_train.replace(row, range(len(row))) 
    #print(adult_train) 
    replace_test = [
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
         'Never-worked'], 
        ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th',
         '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'], 
        ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
         'Married-AF-spouse'],
        ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
         'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
         'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
        ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
        ['Female', 'Male'],
        ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
         'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
         'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
         'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
         'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
        ['<=50K.', '>50K.']
    ]

    for row in replace_test: 
        adult_test = adult_test.replace(row, range(len(row)))
    
    adult_train = adult_train.drop('education', axis=1)
    adult_test = adult_test.drop('education', axis=1)
    #print(adult_train, adult_test)

    adult_train = adult_train.values
    np.random.seed(seed=78712)
    np.random.shuffle(adult_train)
    X_train = adult_train[:, :13].astype(np.float64)
    y_train = adult_train[:, 13].astype(np.uint8)
    
    adult_test = adult_test.values
    X_test = adult_test[:, :13].astype(np.float64)
    y_test = adult_test[:, 13].astype(np.uint8)
    
    X_train, X_test, start_index, cat_length = normalize_data_ours(X_train, X_test)
    
    if get_categorical_info:
        return X_train, y_train, X_test, y_test, start_index, cat_length 
    else:
        return X_train, y_train, X_test, y_test
