import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
Numbers for "os.environ['TF_CPP_MIN_LOG_LEVEL']": 
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import tensorflow
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore Pandas warnings of deprecation
import pandas as pd
import numpy as np
from time import time
from keras.layers import Dense, Dropout, RNN, LSTM, GRU, SimpleRNN
from keras import Sequential, models
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from termcolor import colored

# from kdd_processing import kdd_encoding # not being used
from unsw_processing import unsw_encoding
from results_visualisation import print_results
from convert_lite import tensorflow_to_lite

pd.options.mode.chained_assignment = None  # default='warn' | Disable warnings

# Allows tensorflow to run multiple sessions (Multiple learning simultaneously)
# Comment the 3 following lines if causing issues
# config = tensorflow.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tensorflow.Session(config=config)


csv_values = ['epochs', 'acc', 'loss', 'val_acc', 'val_loss', "train_data",
              "features_nb", 'loss_fct', 'optimizer', 'activation_fct',
              'layer_nb', 'unit_nb', 'batch_size', 'dropout', 'cell_type',
              'encoder']

csv_best_res = ['param', 'value', 'min_mean_val_loss']

# epochs: Number of iteration of the training dataset
# train_data: Number of rows in training dataset (see processing files)(not used in UNSW)
# features_nb: Number of features kept as input (see processing files)(not used in UNSW)
# loss fct: Loss function used in training 
# optimizer: Optimizer used in training
# activation_fct: Activation function used in outputs layer
# layer_nb: Number of hidden layers in the network
# unit_nb: Number of cells for each layer
# batch_size: Number of elements observed before updating weights
# dropout: Fraction of inputs randomly discarded
# cell_type: Type of cell ['RNN', 'LSTM']
# encoder: Encoding performed (see processing files)
# dataset: Processing file to be called ['kdd', 'unsw']
# training_nb: Number of model to be trained with the same params
# resultstocsv: Wether to save results to csv
# resultstologs: Wether to save models and tensorboard logs
# showresults: Wether to show detailled statistics about the trained model
# shuffle: Wether to shuffle the batches sequences during training

# ***** REFERENCES PARAMETERS ***** # Try out different parameters
params = {'epochs': 50, 'train_data': 494021, 'features_nb': 4,
          'loss_fct': 'mse', 'optimizer': 'nadam',
          'activation_fct': 'softmax', 'layer_nb': 2, 'unit_nb': 128,
          'batch_size': 1024, 'dropout': 0.4, 'cell_type': 'RNN',
          'encoder': 'ordinalencoder', 'dataset': 'unsw', 'training_nb': 1,
          'resultstocsv': True, 'resultstologs': True, 'showresults': True,
          'shuffle': True}

# ***** VARIABLE PARAMETERS *****
params_var = {'encoder': ['standardscaler', 'labelencoder', # default = labelencoder
                          'minmaxscaler01', 'minmaxscaler11',
                          'ordinalencoder'],
              'optimizer': ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax', # default = rmsprop
                            'adadelta'],
              'activation_fct': ['sigmoid', 'softmax', 'relu', 'tanh'], # default = sigmoid
              'layer_nb': [1, 2, 3, 4], # default = 1
              'unit_nb': [4, 8, 32, 64, 128, 256], # default = 128
              'dropout': [0.1, 0.2, 0.3, 0.4], # default = 0.2
              'batch_size': [512, 1024, 2048], # default = 1024
              # 'features_nb': [4, 8, 41],
              # 'cell_type': ['RNN', 'LSTM'],
              # 'dataset : ['unsw']
              }

model_path = './models/'
tf_model_path = './tf_models' # Newly added
logs_path = './logs/'
res_path = './results/' + 'testcsv/'

if params['resultstologs'] is True:
    res_name = str(params['train_data']) + '_' + str(params['features_nb']) +\
        '_' + params['loss_fct'] + '_' + params['optimizer'] + '_' +\
        params['activation_fct'] + '_' + str(params['layer_nb']) + '_' +\
        str(params['unit_nb']) + '_' + str(params['batch_size']) + '_' +\
        str(params['dropout']) + '_' + params['cell_type'] + '_' +\
        params['encoder'] + '_' + str(time())


# Encode dataset and return : x_train, x_test, y_train, y_tests
def load_data():
    #if params['dataset'] == 'kdd':
        #x_train, x_test, y_train, y_test = kdd_encoding(params)
    if params['dataset'] == 'unsw':
        x_train, x_test, y_train, y_test = unsw_encoding(params)

    # Reshape the inputs in the accepted model format
    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    return x_train, x_test, y_train, y_test


# Create and train a model
def train_model(x_train, x_test, y_train, y_test):
    if params['cell_type'] == 'RNN':
        cell = SimpleRNN #RNN
    elif params['cell_type'] == 'LSTM':
        cell = LSTM

    if os.path.exists("./tf_models"):
        save_model = ModelCheckpoint(filepath=model_path + res_name,
                                        monitor='val_accuracy', save_best_only=True)
        tensorboard = TensorBoard(logs_path+res_name)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) # Early-Stopping causing issues..
        callbacks = [save_model, tensorboard, es]
        loaded_model = models.load_model('./tf_models')
        loaded_model.fit(x_train, y_train, params['epochs'],
                        verbose=1, shuffle=params['shuffle'],
                        validation_data=(x_test, y_test), callbacks=callbacks)

        # Create model and logs folder if does not exist
        if params['resultstologs'] is True:
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if not os.path.exists(tf_model_path):
                os.makedirs(tf_model_path)

            loaded_model.save(tf_model_path)
            tensorflow_to_lite()

            loaded_model.summary()

            hist = loaded_model.fit(x_train, y_train, params['batch_size'], params['epochs'],
                            verbose=1, shuffle=params['shuffle'],
                            validation_data=(x_test, y_test), callbacks=callbacks)

            if params['showresults'] is True:
                print_results(params, loaded_model, x_train, x_test, y_train, y_test)

            return hist

    else:
        print("\n\n\n{}\n {} \n\n".format(colored("Couldn't find existing model!", 'red'), colored('-- Creating new model --', 'cyan')))
        # Create a Sequential layer, one layer after the other
        model = Sequential()
        # If there is more than 1 layer, the first must return sequences
        for _ in range(params['layer_nb']-1):
            model.add(cell(units=params['unit_nb'],
                        input_shape=(x_train.shape[1:]), return_sequences=True))
            model.add(Dropout(rate=params['dropout']))

        # If there is only 1 layer, it must not return sequences
        if(params['layer_nb'] == 1):
            if params['cell_type'] == "RNN":
                model.add(cell(units=params['unit_nb'], input_shape=x_train.shape[1:]))
                model.add(Dropout(rate=params['dropout']))
            else: 
                model.add(cell(units=params['unit_nb'], input_shape=x_train.shape[1:]))
                model.add(Dropout(rate=params['dropout']))
        else:  # If there is more than 1, the following must not return sequences
            model.add(cell(units=params['unit_nb']))
            model.add(Dropout(rate=params['dropout']))
        # Outputs layer
        model.add(Dense(units=y_train.shape[1],
                        activation=params['activation_fct']))

        model.compile(loss=params['loss_fct'], optimizer=params['optimizer'],
                    metrics=['accuracy'])

        # Create model and logs folder if does not exist
        if params['resultstologs'] is True:
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if not os.path.exists(tf_model_path):
                os.makedirs(tf_model_path)


            save_model = ModelCheckpoint(filepath=model_path + res_name,
                                        monitor='val_accuracy', save_best_only=True)
            tensorboard = TensorBoard(logs_path+res_name)
            callbacks = [save_model, tensorboard]

            model.save(tf_model_path) 
            tensorflow_to_lite()
        else:
            callbacks = None

        model.summary()

        hist = model.fit(x_train, y_train, params['batch_size'], params['epochs'],
                        verbose=1, shuffle=params['shuffle'],
                        validation_data=(x_test, y_test), callbacks=callbacks)

        if params['showresults'] is True:
            print_results(params, model, x_train, x_test, y_train, y_test)

        return hist


def res_to_csv():
    ref_min_val_loss = 10  # Minimal reference loss value
    nsmall = 5  # Number of val loss for the mean val loss

    # Create the results directory if it doesnt exist
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    full_res_path = res_path + 'full_results.csv'
    best_res_path = res_path + 'best_result.csv'

    # Initialize results and best_results dataframes
    results_df = pd.DataFrame(columns=csv_values)
    results_df.to_csv(full_res_path, index=False)

    best_res_df = pd.DataFrame(columns=csv_best_res)

    def fill_dataframe(df, history, epoch):
        try:
            df = df.append({'epochs': epoch,
                            'acc':  history.history['accuracy'][epoch],
                            'loss': history.history['loss'][epoch],
                            'val_acc': history.history['val_accuracy'][epoch],
                            'val_loss': history.history['val_loss'][epoch],
                            'train_data': params['train_data'],
                            'features_nb': params['features_nb'],
                            'loss_fct': params['loss_fct'],
                            'optimizer': params['optimizer'],
                            'activation_fct': params['activation_fct'],
                            'layer_nb': params['layer_nb'],
                            'unit_nb': params['unit_nb'],
                            'batch_size': params['batch_size'],
                            'dropout': params['dropout'],
                            'cell_type': params['cell_type'],
                            'encoder': params['encoder']},
                        ignore_index=True)
            return df
        except Exception as e:
            print("Could not append to df due to {}".format(colored(e, 'red')))

    # Make the mean of the n smallest val_loss for each feature values
    def min_mean_val_loss(feature):
        # Load the results previously saved as csv for the features
        df = pd.read_csv(res_path+feature+".csv", index_col=False)
        names = df[feature].unique().tolist()
        df_loss = pd.DataFrame(columns=names)

        # For each value of the feature, compare the n smallest val loss
        for i in range(len(names)):
            df_value_loss = df.loc[df[feature] == names[i]]
            df_value_loss = df_value_loss.nsmallest(nsmall, 'val_loss')
            df_loss[names[i]] = np.array(df_value_loss['val_loss'])

        # Return the index and the value of the feature
        #  with the smallest mean val loss
        return df_loss.mean().idxmin(), df_loss.mean().min()

    for feature in params_var.keys():
        results_df.to_csv(res_path + feature + ".csv", index=False)
        save_feature_value = params[feature]

        for feature_value in params_var[feature]:
            df_value = pd.DataFrame(columns=csv_values)
            params[feature] = feature_value

            if feature == 'encoder' or feature == 'train_data':
                # The encoding will have to change, so the data are reaload
                x_train, x_test, y_train, y_test = load_data()

            for _ in range(params['training_nb']):
                history = train_model(x_train, x_test, y_train, y_test)

                # The datafranme is filled for each epoch
                for epoch in range(params['epochs']):
                    df_value = fill_dataframe(df_value, history, epoch)
            # At the end of the training, results are saved in csv
            try:
                df_value.to_csv(full_res_path, header=False, index=False, mode='a')
                df_value.to_csv(res_path + feature + ".csv", header=False,
                            index=False, mode='a')
            except Exception as e:
                print("Could not save result due to {}".format(colored(e, 'red')))
        # Once the test of the value is over, return the min mean val loss
        feature_value_min_loss, min_mean_loss = min_mean_val_loss(feature)

        # Compare the best val loss for the feature value with the reference
        # of the best val loss. If better, the best val becomes the reference,
        # and feature value correspondind is chosen for the rest of the test
        if min_mean_loss < ref_min_val_loss:
            params[feature] = feature_value_min_loss
            ref_min_val_loss = min_mean_loss
        else:
            params[feature] = save_feature_value

        # Save the best feature value, reference is saved if better
        best_res_df = best_res_df.append({'param': feature,
                                          'value': params[feature],
                                          'min_mean_val_loss': min_mean_loss},
                                         ignore_index=True)
        best_res_df.to_csv(best_res_path, index=False)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()

    for i in range(params['training_nb']):
        if params['resultstocsv'] is False:
            train_model(x_train, x_test, y_train, y_test)
        else:
            res_to_csv()
            