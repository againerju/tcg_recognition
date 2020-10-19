from __future__ import print_function
import os
import configparser

import numpy as np
import tensorflow as tf
from keras import backend as K
import random as rn
from time import strftime

import TCGDB
import train

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'config', 'config.ini'))

os.environ['PYTHONHASHSEED'] = str(config['seed']['python_seed'])
np.random.seed(int(config['seed']['np_seed']))
tf.set_random_seed(int(config['seed']['tf_seed']))
rn.seed(int(config['seed']['random_seed']))

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)


def experiment_mkdir(config):

    if not os.path.isdir(config['path']['run_root']):
        os.makedirs(config['path']['run_root'])

    filename = config["model"]['name'].lower() + '_' + strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(config['path']['run_root'], filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def main(config):
    
    ##### EXPERIMENT #####
    savedir = experiment_mkdir(config)

    ##### DATASET #####
    db = TCGDB.TCGDB(config["path"]["tcg_root"])

    db.open()

    db.set_subclasses(train_subclasses=int(config["training-mode"]["subclasses"]))

    print("Finished opening TCG dataset.")

    ##### LEARNING #####
    for run_id in range(db.get_number_runs(protocol_type=config["training-mode"]["protocol_type"])):

        X_train, Y_train, X_test, Y_test = db.get_train_test_data(run_id=run_id,
                                                                  protocol_type=config["training-mode"]["protocol_type"])

        print("\n{:10}: ".format("Run"), db.get_train_test_set(run_id=run_id,
                                                               protocol_type=config["training-mode"]["protocol_type"]))
        print("{:10}: ".format("X_train"), X_train.shape, " | {:10}: ".format("Y_train"), Y_train.shape)
        print("{:10}: ".format("X_test"), X_test.shape, " | {:10}: ".format("Y_test"), Y_test.shape)

        model = train.fit_model(config, X_train, Y_train, X_test, Y_test, savedir, str(run_id))

        train.evaluate_model(config, model, X_test, Y_test, savedir, str(run_id))


if __name__ == "__main__":
    main(config)
