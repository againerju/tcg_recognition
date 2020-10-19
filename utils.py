from __future__ import print_function

import os
import json
import h5py 
import numpy as np
import itertools

import keras
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


# =============================================================================
# SEQUENCE DATA
# =============================================================================

def get_labels(config):
    if config["training-mode"]["subclasses"]:
        label_names = ["idle", 'stop_both-static', 'stop_both-dynamic', 'stop_left-static', 'stop_left-dynamic',
                       'stop_right-static', 'stop_right-dynamic', 'clear_left-static', 'clear_right-static',
                       'go_both-static', 'go_both-dynamic', 'go_left-static', 'go_left-dynamic', 'go_right-static',
                       'go_right-dynamic']
    else:
        label_names = ["idle", "stop", "go", "clear"]

    return label_names


# =============================================================================
# SEQUENCE POSTPROCESSING
# =============================================================================

def read_h5_file(filename):
    with h5py.File(filename, 'r') as f:
        file_key = list(f.keys())[0]
        data = list(f[file_key])
    return data


def read_json(filename):
    with open(filename, 'r') as myfile:
        data_db = myfile.read()
    dict_db = json.loads(data_db)
    return dict_db


def OneHotEncoding(label_seq, subclasses=False):
    if subclasses:
        labels_enc = np.zeros([len(label_seq), 15])
        for i in range(len(label_seq)):
            if label_seq[i] == 'idle_normal-pose' or label_seq[i] == 'idle_out-of-vocabulary' or label_seq[i] == 'idle_transition':
                labels_enc[i, 0] = 1
            if label_seq[i] == 'stop_both-static' :
                labels_enc[i, 1] = 1
            if label_seq[i] == 'stop_both-dynamic' :
                labels_enc[i, 2] = 1
            if label_seq[i] == 'stop_left-static' :
                labels_enc[i, 3] = 1
            if label_seq[i] == 'stop_left-dynamic' :
                labels_enc[i, 4] = 1
            if label_seq[i] == 'stop_right-static':
                labels_enc[i, 5] = 1
            if label_seq[i] == 'stop_right-dynamic':
                labels_enc[i, 6] = 1
            if label_seq[i] == 'clear_left-static':
                labels_enc[i, 7] = 1
            if label_seq[i] == 'clear_right-static':
                labels_enc[i, 8] = 1
            if label_seq[i] == 'go_both-static' :
                labels_enc[i, 9] = 1
            if label_seq[i] == 'go_both-dynamic':
                labels_enc[i, 10] = 1
            if label_seq[i] == 'go_left-static':
                labels_enc[i, 11] = 1
            if label_seq[i] == 'go_left-dynamic':
                labels_enc[i, 12] = 1
            if label_seq[i] == 'go_right-static':
                labels_enc[i, 13] = 1
            if label_seq[i] == 'go_right-dynamic':
                labels_enc[i, 14] = 1
    else:
        labels_enc = np.zeros([len(label_seq), 4])
        for i in range(len(label_seq)):
            if label_seq[i] == 'idle':
                labels_enc[i, 0] = 1
            if label_seq[i] == 'stop':
                labels_enc[i, 1] = 1
            if label_seq[i] == 'go':
                labels_enc[i, 2] = 1
            if label_seq[i] == 'clear':
                labels_enc[i, 3] = 1

    return labels_enc

# =============================================================================
# SEQUENCE POSTPROCESSING
# =============================================================================

def OneHotDecoding(labels, subclasses=False):
    labels_dec = [None] * labels.shape[0]

    for i in range(len(labels_dec)):
        if subclasses:
            if labels[i] == 0:
                labels_dec[i] = 'idle'
            if labels[i] == 1:
                labels_dec[i] = 'stop_both-static'
            if labels[i] == 2:
                labels_dec[i] = 'stop_both-dynamic'
            if labels[i] == 3:
                labels_dec[i] = 'stop_left-static'
            if labels[i] == 4:
                labels_dec[i] = 'stop_left-dynamic'
            if labels[i] == 5:
                labels_dec[i] = 'stop_right-static'
            if labels[i] == 6:
                labels_dec[i] = 'stop_right-dynamic'
            if labels[i] == 7:
                labels_dec[i] = 'clear_left-static'
            if labels[i] == 8:
                labels_dec[i] = 'clear_right-static'
            if labels[i] == 9:
                labels_dec[i] = 'go_both-static'
            if labels[i] == 10:
                labels_dec[i] = 'go_both-dynamic'
            if labels[i] == 11:
                labels_dec[i] = 'go_left-static'
            if labels[i] == 12:
                labels_dec[i] = 'go_left-dynamic'
            if labels[i] == 13:
                labels_dec[i] = 'go_right-static'
            if labels[i] == 14:
                labels_dec[i] = 'go_right-dynamic'
        else:
            if labels[i] == 0:
               labels_dec [i] = 'idle'
            if labels[i] == 1:
                labels_dec [i] = 'stop'
            if labels[i] == 2:
                labels_dec [i] = 'go'
            if labels[i] == 3:
                labels_dec [i] = 'clear'

    return labels_dec
    

def binarize_predictions(test_data, true_label, predictions, subclasses=False):
    """
    make model predictions binary 
   
    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_timesteps, n_classes)
        Probability estimates of the class labels
    true_label : array-like of shape (n_samples, n_timesteps, n_classes)
        True class labels
    test_data : array-like of shape (n_samples, n_timesteps, n_features)
        Input data
        
    Returns
    -------
    test_data : array-like of shape (n_samples, n_timesteps, n_features)
        Input data
    true_classes : array-like of shape (n_samples, n_timesteps, n_classes)
        OneHotDecoding of the true classes
        
    """
    
    indices = np.where(~test_data.any(axis=1))[0]   
    predictions = np.delete(predictions, indices, axis=0)      
    true_label = np.delete(true_label, indices, axis=0)    
    test_data = np.delete(test_data, indices, axis=0)       
    pred_classes = OneHotDecoding (np.argmax(predictions, axis=1), subclasses)
    true_classes = OneHotDecoding (np.argmax(true_label, axis=1), subclasses)
    return test_data, true_classes, pred_classes


def delete_pading(test_data, true_label, predictions):
    indices = np.where(~test_data.any(axis=1))[0]   
    predictions = np.delete(predictions, indices, axis=0)      
    true_label = np.delete(true_label, indices, axis=0)    
    test_data = np.delete(test_data, indices, axis=0)       
    return test_data, true_label,  predictions


  
#%% Model development
     
# =============================================================================
# MODEL TRAINING AND PREDICTIONS
# =============================================================================
  
def evaluate_training(y_score, y_test, history):
    
    """
    plot validation and training curves
   
    Parameters
    ----------
    y_score : array-like of shape (n_samples, n_timesteps, n_classes)
        Probability estimates of the class labels
    y_test : array-like of shape (n_samples, n_timesteps, n_classes)
        True class labels
    history: History Object
        
    """

    plt.figure(figsize=(10, 10))
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training','validation'], loc='best') 
    plt.savefig('model accuracy')
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='best')
    plt.savefig('model loss')
    plt.show()


def save_model(model, filename):
    
    """
    Serlailize model to JSON and save weights
   
    Parameters
    ----------
    model : trained tensorflow model
    filename: Name of the model
        
    """
    # 
    model_json = model.to_json()
    filename_json = filename + ".json"
    with open(filename_json, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        filename_h5 = filename + ".h5"
        model.save_weights(filename_h5)
        print("Saved model to disk")


def load_trained_model (model_name):
    
    """
   load a pretrained neural network
   
    Parameters
    ----------
    model path : str
        path of the pretrained model

    Returns
    -------
    pretrained model
        
    """
    json_file = open(os.path.join(model_name + '.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    trained_model = keras.models.model_from_json(loaded_model_json)
    trained_model.load_weights(os.path.join(model_name + '.h5'))   
    return trained_model


def load_model_attention(model_path):
    """
   load a pretrained recurrent network with attention mechanism
   
    Parameters
    ----------
    model path : str
        path of the pretrained attention model 

    Returns
    -------
    pretrained model with attention mechanism
        
     """

    from keras_self_attention import SeqSelfAttention
    import keras
    json_file = open(os.path.join(model_path + '.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    trained_model = keras.models.model_from_json(loaded_model_json,custom_objects=SeqSelfAttention.get_custom_objects())
    # load weights into new model
    trained_model.load_weights(os.path.join(model_path + '.h5'))
    print("trained model loaded")
    
    model = trained_model
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
        
    return model


def load_model_TCN(model_path):
    
    """
   load a pretrained Temporal Convolutional neural network
   
    Parameters
    ----------
    model path : str
        path of the pretrained model

    Returns
    -------
    pretrained TCN model
        
     """
    
    from tcn import TCN
    from keras.models import model_from_json
    
    json_file = open(os.path.join(model_path + '.json'), 'r')
    loaded_model_json = json_file.read()
    reloaded_model = model_from_json(loaded_model_json, custom_objects={'TCN': TCN})
    reloaded_model.load_weights(os.path.join(model_path + '.h5'))
    print("trained TCN loaded")
    reloaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return reloaded_model


def evaluate_trained_model(model_name, X_test, Y_test):
    """
    evaluate a trained tf-model.
    
    Parameters
    ----------
    model_name : str
    y_test : array-like of shape (n_samples, n_timesteps, n_classes)
    X_test: array-like of shape (n_samples, n_timesteps, n_features)
    
    classes : array of srt of shape (n_classes,)
        Probability estimates of the class labels
    
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized

   
    """
    
    model = load_trained_model(model_name)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.summary()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#%% Model evluation
    
# =============================================================================
# CONFUSION MATRIX FOR MULTI-CLASS CLASSIFICATION
# =============================================================================    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Print and plot the confusion matrix.
    
    Refrence Code: https://github.com/scikit-learn/scikit-learn/sklearn/metrics
    
    Parameters
    ----------
    cm : ndarray of shape (n_classes, n_classes)
    
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and prediced label being j-th class.

    classes : array of srt of shape (n_classes,)
        Probability estimates of the class labels
    
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, y=1.08)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):                   
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# =============================================================================
# ROC CURVE FOR MULTI-CLASS CLASSIFICATION
# =============================================================================

def plot_roc_multiclass(y_test, y_score, n_classes, labels):
    """
    Plot Receiver operating characteristics for multiclass classification.
    Reference code -https://github.com/scikit-learn/scikit-learn/sklearn/metrics/

    Parameters
    ----------
    y_test : array-like of shape (n_samples, n_classes)
        True Labels
    y_score : array-like of shape (n_samples, n_classes)
        Probability estimates of the class labels

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
        
     """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    #  average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(20, 10))
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue',
                              'forestgreen', 'red', 'darkcyan', 'blue'])
    lw = 2
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    return fpr, tpr


# =============================================================================
# SAVE TRAINING AND TEST DATA TO HDF5
# =============================================================================    
    
def save_train_test(X_train, X_test, Y_train, Y_test):
    
    """

    Args:
        sequence: 2-d array (time x features)
        max_len: int

    Returns:
        pad_seq: zero padded sequence (max_len X features)

    """
    h5f = h5py.File('train.h5', 'w')
    h5f.create_dataset('X_train', data=X_train)

    h5f.create_dataset('Y_train', data=Y_train)
    h5f.close()


    h5f = h5py.File('test.h5', 'w')
    h5f.create_dataset('X_test', data=X_test)

    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()

    
def read_saved_data():
    
    h5f = h5py.File('train.h5', 'r')
    X_train = h5f['X_train'][:]
    Y_train = h5f['Y_train'][:]
    h5f.close()

    h5f = h5py.File('test.h5','r')
    X_test = h5f['X_test'][:]
    Y_test = h5f['Y_test'][:]
    h5f.close()
    
    return X_train, Y_train, Y_test, X_test


# =============================================================================
# TCGDB UTILITY FUNCTIONS
# =============================================================================

def pad_sequence(sequence, max_len):
    """

    Args:
        sequence: 2-d array (time x features)
        max_len: int

    Returns:
        pad_seq: zero padded sequence (max_len X features)

    """

    # validity check
    assert sequence.shape[0] <= max_len

    # zero padding
    pad_seq = np.zeros((max_len, sequence.shape[1]))
    pad_seq[0:sequence.shape[0], :] = sequence

    return pad_seq


def one_hot_encoding(targets, nb_classes):
    """

    Args:
        targets: target sequence [n_samples]
        nb_classes: int

    Returns:
        targets_one_hot: one-hot-encoded targets [n_samples x nb_classes]

    """

    targets_one_hot = np.eye(nb_classes)[targets]

    return targets_one_hot

def subsampling(sequence, sampling_factor=5):
    """

    Args:
        sequence: data sequence [n_samples, n_features]
        sampling_factor: sub-sampling factor

    Returns:
        sequence: sub-sampled data sequence [n_samples/sampling_factor, n_features]

    """

    sequence = sequence[::sampling_factor]

    return sequence
