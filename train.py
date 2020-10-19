from __future__ import print_function
import numpy as np
import os

from keras.layers import Dense, GRU, Bidirectional, LSTM, SimpleRNN
from keras.models import Sequential
from keras.layers import Dropout
from keras_self_attention import SeqSelfAttention

from keras.optimizers import Adam
from tcn import TCN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import utils


# =============================================================================
# model training and validation    
# =============================================================================

def fit_model(config, X_train, Y_train, X_test, Y_test, savedir, combination):

    model = Sequential()
    
    for i in range(int(config[config["model"]["name"]]['hidden_layers'])):

        if config["model"]["name"] == "RNN":
            model.add(SimpleRNN(units=int(config[config["model"]["name"]]['n_cells']),
                                return_sequences=True)) 
            model.add (Dropout(float(config[config["model"]["name"]]['dropout_rate'])))
        if config["model"]["name"] == "LSTM":
            model.add(LSTM(units=int(config[config["model"]["name"]]['n_cells']),
                           return_sequences=True)) 
            model.add(Dropout(float(config[config["model"]["name"]]['dropout_rate'])))
        if config["model"]["name"] == "att_LSTM":
            model.add(LSTM(units=int(config[config["model"]["name"]]['n_cells']),
                           return_sequences=True)) 
            model.add(SeqSelfAttention(int(config[config["model"]["name"]]['attention_units']),
                                       attention_activation='sigmoid')) 
            model.add (Dropout(float(config[config["model"]["name"]]['dropout_rate'])))
        if config["model"]["name"] == "GRU":
            model.add(GRU(units=int(config[config["model"]["name"]]['n_cells']),
                          return_sequences=True)) 
            model.add (Dropout(float(config[config["model"]["name"]]['dropout_rate'])))
        if config["model"]["name"] == "Bi_LSTM":
            model.add(Bidirectional(LSTM(units=int(config[config["model"]["name"]]['n_cells']),
                                         return_sequences=True)))
            model.add (Dropout(float(config[config["model"]["name"]]['dropout_rate'])))
        if config["model"]["name"] == "Bi_GRU":
            model.add(Bidirectional(GRU(units=int(config[config["model"]["name"]]['n_cells']),
                                        return_sequences=True)))  
            model.add (Dropout(float(config[config["model"]["name"]]['dropout_rate'])))
        if config["model"]["name"] == "TCN":
            model.add(TCN(nb_filters=int(config[config["model"]["name"]]['nb_filters']),
                          kernel_size=int(config[config["model"]["name"]]['kernel_size']),
                          nb_stacks=int(config[config["model"]["name"]]['nb_stacks']),
                          dropout_rate=float(config[config["model"]["name"]]['dropout_rate']),
                          dilations=(1, 2), activation='relu', return_sequences=True,
                          use_skip_connections=True, padding='causal',
                          use_batch_norm=False,
                          input_shape=(None, X_train.shape[2])))
                           
    if int(config['training-mode']['subclasses']) == 1:
        model.add(Dense(units=int(config[config["model"]["name"]]['minor_classes']),
                        activation='softmax'))     
    else:
        model.add(Dense(units=int(config[config["model"]["name"]]["major_classes"]),
                        activation='softmax'))

    opt = Adam(lr=float(config[config["model"]["name"]]["learning_rate"]))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    history = model.fit(X_train, Y_train,
                        epochs=int(config[config["model"]["name"]]["n_epochs"]),
                        batch_size=int(config[config["model"]["name"]]["batch_size"]),
                        validation_data=(X_test, Y_test),
                        verbose=int(config[config["model"]["name"]]["verbose"]))

    model.summary()
    
    scores = model.evaluate(X_test, Y_test)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0])) 
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    plt.figure(figsize=(10,10))
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training','validation'], loc='best') 
    plt.savefig(os.path.join(savedir,'model accuracy'+'_'+combination))

    plt.figure(figsize=(10,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['model loss', 'validation loss'], loc='best') 
    plt.savefig(os.path.join(savedir,'model loss'+'_'+combination))

    utils.save_model(model, os.path.join(savedir, os.path.basename(savedir)+'_'+combination))
    
    return model


def evaluate_model (config,model,X_test,Y_test,savedir,combination):
    features = X_test.shape[2]
    targets = Y_test.shape[2]
    
    major_classes = ['idle', 'stop', 'go', 'clear']
    
    minor_classes = ["idle",
                     "stop_both-static", "stop_both-dynamic", "stop_left-static",
                     "stop_left-dynamic", "stop_right-static", "stop_right-dynamic",
                     "clear_left-static", "clear_right-static",
                     "go_both-static", "go_both-dynamic", "go_left-static",
                     "go_left-dynamic", "go_right-static", "go_right-dynamic"]
    
    predictions = model.predict(X_test)

    
    if int (config ['training-mode']['subclasses']) == 1:
        classes_tcg = minor_classes
        predictions_bin = utils.binarize_predictions(X_test.reshape(-1, features),
                                                     Y_test.reshape(-1, targets),
                                                     predictions.reshape(-1, targets),
                                                     subclasses=True)
    else:
        classes_tcg = major_classes

        predictions_bin = utils.binarize_predictions(X_test.reshape(-1, features),
                                                     Y_test.reshape(-1, targets),
                                                     predictions.reshape(-1, targets))

    cnf_matrix = confusion_matrix(predictions_bin[1], predictions_bin[2], labels=classes_tcg)
    
    np.set_printoptions(precision=2)  
    plt.figure(figsize=(30, 12))
    plt.subplot(121)
    utils.plot_confusion_matrix(cnf_matrix, classes=classes_tcg, title='Confusion matrix, without normalization')
    
    plt.close('all')
    plt.subplot(122)
    utils.plot_confusion_matrix(cnf_matrix, classes=classes_tcg, normalize=True,
                                title='Confusion matrix with normalization')

    plt.suptitle("confusion matrix")
    plt.subplots_adjust(top=0.88)

    plt.savefig(os.path.join(savedir, 'cm' + '_' + combination))
    
    unpadded_seq = utils.delete_pading(X_test.reshape(-1, X_test.shape[2]),
                                       Y_test.reshape(-1, targets),
                                       predictions.reshape(-1, targets))
    
    utils.plot_roc_multiclass(unpadded_seq[1], unpadded_seq[2],
                              targets, classes_tcg)
        
    plt.savefig(os.path.join(savedir, 'roc'+'_'+combination))
    
    plt.close('all')
