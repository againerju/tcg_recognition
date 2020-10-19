from __future__ import print_function

import os
import configparser
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'config', 'config.ini'))

import numpy as np
import utils

from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import TCGDB


def test_model(config, model_name):
    
    db = TCGDB.TCGDB(config["path"]["tcg_root"])

    db.open()

    db.set_subclasses(train_subclasses=int(config["training-mode"]["subclasses"]))

    print("Finished opening TCG dataset.")
    
    for run_id in range(db.get_number_runs(protocol_type=config["training-mode"]["protocol_type"])):

        X_train, Y_train, X_test, Y_test = db.get_train_test_data(run_id=run_id,
                                                                  protocol_type=config["training-mode"]["protocol_type"])

        if config["model"]["name"] == 'TCN':
            model = utils.load_model_TCN(os.path.join(model_name, "combination_{}".format(run_id+1)))
        else:
            model = utils.load_trained_model(os.path.join(model_name, "combination_{}".format(run_id+1)))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

        predictions = model.predict(X_test)
            
        if int(config['training-mode']['subclasses']) == 1:
            y_test_arr = Y_test.reshape(-1, 4)
            preds_arr = predictions.reshape(-1, 4)
        else:
            y_test_arr = Y_test.reshape(-1, 15)
            preds_arr = predictions.reshape(-1, 15)
            
        non_zero_indices = np.where(np.sum(y_test_arr, axis=1) != 0)[0]
        
        y_test_wo_pad_ohenc = y_test_arr[non_zero_indices]
        preds_wo_pad_ohenc = preds_arr[non_zero_indices]

        y_test_eval = np.argmax(y_test_wo_pad_ohenc, axis=1)
        preds_eval = np.argmax(preds_wo_pad_ohenc, axis=1)

        acc = accuracy_score(y_test_eval, preds_eval)

        predictions_bin = utils.binarize_predictions(X_test.reshape(-1, X_test.shape[2]),
                                                     Y_test.reshape(-1, Y_test.shape[2]),
                                                     predictions.reshape(-1, Y_test.shape[2]), 
                                                     subclasses= bool (int(config['training-mode']["subclasses"])))
        
        cnf_matrix = confusion_matrix(predictions_bin[1], predictions_bin[2], labels=utils.get_labels(config))

        jaccard_index = jaccard_score(np.argmax(Y_test, axis=2).reshape(-1),
                                      np.argmax(predictions, axis=2).reshape(-1), average='macro')

        f1 = f1_score(np.argmax(Y_test, axis=2).reshape(-1),
                      np.argmax(predictions, axis=2).reshape(-1), average='macro')
        
        return acc, cnf_matrix, jaccard_index, f1


def get_model_directory(config):

    baseline_root = config["path"]["baseline_root"]
    model_name = config["model"]["name"].lower()
    protocol_type = config["training-mode"]["protocol_type"]
    major_minor = {"0": "major", "1": "minor"}[config["training-mode"]["subclasses"]]
    model_directory = os.path.join(baseline_root, model_name + "_" + protocol_type + "_" + major_minor)

    return model_directory


if __name__ == "__main__":

    baseline_path = get_model_directory(config)

    performance = test_model(config, baseline_path)

    print("Evaluation done.")
