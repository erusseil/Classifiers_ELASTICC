# Copyright 2020-2021 
# Author: Marco Leoni and Emille Ishida
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import numpy as np
import os

from actsnclass import DataBase
import warnings
import timeit

import actsnclass

warnings.filterwarnings("ignore", category=RuntimeWarning)

def learn_loop(data: actsnclass.DataBase, nloops: int, strategy: str,
               output_metrics_file: str, output_queried_file: str,
               classifier='RandomForest', batch=1, screen=True, nest=100):
    """Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    nloops: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    path_to_features: str
        Complete path to input features file.
    output_metrics_file: str
        Full path to output file to store metric values of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    features_method: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently only 'RandomForest' is implemented.
    training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    """

    for loop in range(nloops):

        if screen:
            print('Processing... ', loop)

        # classify
        data.classify(method=classifier, n_est=nest)

        # calculate metrics
        data.evaluate_classification()

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch)

        # update training and test samples
        data.update_samples(indx, loop=loop)

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=loop)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop, batch=batch,
                                 full_sample=False)


def build_samples(features: pd.DataFrame, initial_training: int,
                 frac_target=0.5, screen=False):
    """Build initial samples for Active Learning loop.
    
    Parameters
    ----------
    features: pd.DataFrame
        Complete feature matrix
    initial_training: int
        Number of objects in the training sample.
    frac_target: float (optional)
        Fraction of target in training. Default is 0.5.
    screen: bool (optional)
        If True, print intermediary information to screen.
        Default is False.
        
    Returns
    -------
    actsnclass.DataBase
        DataBase for active learning loop
    """
    data = DataBase()
    
    # initialize the temporary label holder
    train_indexes = np.random.choice(np.arange(0, features.shape[0]),
                                     size=initial_training, replace=False)
    
    target_flag = features['type'].values == 1
    target_indx = np.arange(0, features.shape[0])[target_flag]
    nontarget_indx =  np.arange(0, features.shape[0])[~target_flag]
    
    indx_target_choice = np.random.choice(target_indx, size=max(1, initial_training // 2),
                                      replace=False)
    indx_nontarget_choice = np.random.choice(nontarget_indx, 
                        size=initial_training - max(1, initial_training // 2),
                        replace=False)
    train_indexes = list(indx_target_choice) + list(indx_nontarget_choice)
    
    temp_labels = features['type'].values[np.array(train_indexes)]

    if screen:
        print('\n temp_labels = ', temp_labels, '\n')

    # set training
    train_flag = np.array([item in train_indexes for item in range(features.shape[0])])
    
    train_target_flag = features['type'].values[train_flag] == 1
    data.train_labels = train_target_flag.astype(int)
    data.train_features = features[train_flag].values[:,2:]
    data.train_metadata = features[['id', 'type']][train_flag]
    
    # set test set as all objs apart from those in training
    test_indexes = np.array([i for i in range(features.shape[0])
                             if i not in train_indexes])
    test_target_flag = features['type'].values[test_indexes] == 1
    data.test_labels = test_target_flag.astype(int)
    data.test_features = features[~train_flag].values[:, 2:]
    data.test_metadata = features[['id', 'type']][~train_flag]
    
    # set metadata names
    data.metadata_names = ['id', 'type']
    
    # set everyone to queryable
    data.queryable_ids = data.test_metadata['id'].values
    
    if screen:
        print('Training set size: ', data.train_metadata.shape[0])
        print('Test set size: ', data.test_metadata.shape[0])
        print('  from which queryable: ', len(data.queryable_ids))
        
    return data


def read_initial_samples(fname_train: str, fname_test:str):
    """Read initial training and test samples from file. 
    
    Parameters
    ----------
    fname_train: str
        Full path to training sample file.
    fname_test: str
        Full path to test sample file.
        
    Returns
    -------
    actsnclass.DataBase
        DataBase for active learning loop.    
    """
    
    # read data    
    data_train = pd.read_csv(fname_train)
    data_test = pd.read_csv(fname_test)

    # build DataBase object
    data = DataBase()
    data.metadata_names = ['id', 'type']
    
    data.train_labels = data_train.values[:,-1] == 1
    data.train_features = data_train.values[:,:-2]
    data.train_metadata = data_train[['objectId', 'type']]
    data.train_metadata.rename(columns={'objectId':'id'}, inplace=True)
    
    data.test_labels = data_test.values[:,-1] == 1
    data.test_features = data_test.values[:,:-2]
    data.test_metadata = data_test[['objectId', 'type']]
    data.test_metadata.rename(columns={'objectId':'id'}, inplace=True)
    
    data.queryable_ids = data.test_metadata[data.metadata_names[0]].values
    
    return data


def main(core_index, job_per_core):

    ################################################################
    
    #########     User choices: general    #########################   
    
    file_name = 'train_AGN'
    fname_features_matrix = f'input/{file_name}.parquet'   # output features file
    dirname_output = f'results/{file_name}'                        # root products output directory
    append_name = ''                                  # append to all metric, prob and queries names
    
    nloops = 5000                   # number of learning loops
    nest = 25                      # n trees
    batch_size = 10
    strategy = 'RandomSampling'            # query strategy
    initial_training = 100                # total number of objs in initial training
    #frac_target_tot = 0.5                   # fraction of target in initial training\\
    already_computed = 0
    n_realizations_ini = core_index*job_per_core + already_computed             # start from this realization number
    n_realizations = job_per_core       # total number of realizations
    
    
    drop_zeros = False                  # ignore objects with observations in only 1 filter
    screen = False                       # print debug comments to screen
    
    features_names = list(pd.read_parquet(fname_features_matrix).keys())
    features_names.remove('id')
    features_names.remove('type')
    
    
    #####  User choices: For Figure 7      ##########################
    
    initial_state_from_file = False      # read initial state from a fixed file
    initial_state_version = 68            # version from which initial state is chosen
    
    ################################################################
    ################################################################
    

    for name in [dirname_output + '/', 
                 dirname_output + '/data/', 
                 dirname_output + '/' + strategy + '/', 
                 dirname_output + '/' + strategy + '/class_prob/',
                 dirname_output + '/' + strategy + '/metrics/', 
                 dirname_output + '/' + strategy + '/queries/',
                 dirname_output + '/' + strategy + '/training_samples/', 
                 dirname_output + '/' + strategy + '/test_samples/']:
        if not os.path.isdir(name):
            os.makedirs(name)  
    
    matrix_clean = pd.read_parquet(fname_features_matrix)    
    
    if initial_state_from_file:
        fname_ini_train = dirname_output + '/UncSampling/training_samples/initialtrain_v' + str(initial_state_version) + '.csv'              
        fname_ini_test = dirname_output + '/UncSampling/test_samples/initial_test_v' + str(initial_state_version) + '.csv'
    
        output_metrics_file = dirname_output + '/' + strategy + '/metrics/metrics_' + strategy + '_v' + str(initial_state_version) + append_name + '.dat'
        output_queried_file = dirname_output + '/' + strategy + '/queries/queried_' + strategy + '_v'+ str(initial_state_version) + append_name + '.dat'
        output_prob_root = dirname_output + '/' + strategy + '/class_prob/v' + str(initial_state_version) + '/class_prob_' + strategy + append_name
    
        name = dirname_output + '/' + strategy + '/class_prob/v' + str(initial_state_version) + '/'
        if not os.path.isdir(name):
            os.makedirs(name)
        data = read_initial_samples(fname_ini_train, fname_ini_test)
        
        # perform learnin loop
        learn_loop(data, nloops=nloops, strategy=strategy, 
                   output_metrics_file=output_metrics_file, 
                   output_queried_file=output_queried_file,
                   classifier='RandomForest', batch=batch_size, screen=True, nest=nest)
        
    else:
        for v in range(n_realizations_ini, n_realizations_ini+n_realizations):
            output_metrics_file = dirname_output + '/' + strategy + '/metrics/metrics_' + strategy + '_v' + str(v) + append_name + '.dat'
            output_queried_file = dirname_output + '/' + strategy + '/queries/queried_' + strategy + '_v'+ str(v) + append_name + '.dat'
            output_prob_root = dirname_output + '/' + strategy + '/class_prob/v' + str(v) + '/class_prob_' + strategy + append_name
    
            name = dirname_output + '/' + strategy + '/class_prob/v' + str(v) + '/'
            if not os.path.isdir(name):
                os.makedirs(name)
            #build samples        
            data = build_samples(matrix_clean, initial_training=initial_training, screen=True)
        
            # save initial data        
            train = pd.DataFrame(data.train_features, columns=features_names)
            train['objectId'] = data.train_metadata['id'].values
            train['type'] = data.train_metadata['type'].values
            train.to_csv(dirname_output + '/' + strategy + '/training_samples/initialtrain_v' + str(v) + '.csv', index=False)
        
            test = pd.DataFrame(data.test_features, columns=features_names)
            test['objectId'] = data.test_metadata['id'].values
            test['type'] = data.test_metadata['type'].values
            test.to_csv(dirname_output + '/' + strategy + '/test_samples/initial_test_v' + str(v) + '.csv', index=False)        
    
            # perform learnin loop
            learn_loop(data, nloops=nloops, strategy=strategy, 
                       output_metrics_file=output_metrics_file, 
                       output_queried_file=output_queried_file,
                       classifier='RandomForest', batch=batch_size, screen=True, nest=nest)
    
if __name__ == '__main__':
    main()
