import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

config['seed'] = {'tf_seed': 1234,
                  'random_seed': 12345,
                  'np_seed': int(42),
                  'python_seed': 0}

project_root = os.path.join(os.getcwd(), '..')

config['path'] = {'tcg_root': project_root,
                  'baseline_root': os.path.join(project_root, "baselines"),
                  'run_root':  os.path.join(project_root, "runs")
                  }

config['training-mode'] = {'protocol_type': 'xv',
                           'subclasses': 0}

config["model"] = {"name": 'TCN'}

config['RNN'] = {'mask_value': 0,
                 'n_cells': 100,
                 'hidden_layers': 1,
                 'n_epochs': 70,
                 'batch_size': 100,
                 'major_classes': 4,
                 'minor_classes': 15,
                 'optimizer': 'adam',
                 'learning_rate': 0.001,
                 'dropout_rate': 0.5,
                 'verbose': 2}

config['GRU'] = {'mask_value': 0,
                 'n_cells': 100,
                 'hidden_layers': 1,
                 'n_epochs': 70,
                 'batch_size': 100,
                 'major_classes': 4,
                 'minor_classes': 15,
                 'optimizer': 'adam',
                 'learning_rate': 0.001,
                 'dropout_rate': 0.5,
                 'verbose': 2,
                 'attention_units': 0}

config['LSTM'] = {'mask_value': 0,
                  'n_cells': 100,
                  'hidden_layers': 1,
                  'n_epochs': 70,
                  'batch_size': 100,
                  'major_classes': 4,
                  'minor_classes': 15,
                  'optimizer': 'adam',
                  'learning_rate': 0.0001,
                  'dropout_rate': 0.5,
                  'verbose': 2}

config['Bi_GRU'] = {'mask_value': 0,
                    'n_cells': 100,
                    'hidden_layers': 1,
                    'n_epochs': 200,
                    'batch_size': 100,
                    'major_classes': 4,
                    'minor_classes': 15,
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'dropout_rate': 0.5,
                    'verbose': 2}

config['Bi_LSTM'] = {'mask_value': 0,
                     'n_cells': 100,
                     'hidden_layers': 1,
                     'n_epochs': 2,
                     'batch_size': 100,
                     'major_classes': 4,
                     'minor_classes': 15,
                     'optimizer': 'adam',
                     'learning_rate': 0.001,
                     'dropout_rate': 0.5,
                     'verbose': 2}

config['att_LSTM'] = {'mask_value': 0,
                      'n_cells': 50,
                      'hidden_layers': 1,
                      'n_epochs': 50,
                      'batch_size': 100,
                      'major_classes': 4,
                      'minor_classes': 15,
                      'optimizer': 'adam',
                      'learning_rate': 0.001,
                      'attention_units': 50,
                      'dropout_rate': 0.5,
                      'verbose': 2}

config['TCN'] = {'mask_value': 0,
                 'dilation_rate': [1, 2, 4, 8, 16, 32], 
                 'nb_filters': 64,
                 'nb_stacks':1,
                 'kernel_size': 2,
                 'hidden_layers': 1,
                 'n_epochs': 5000, 
                 'batch_size': 100,
                 'major_classes': 4,
                 'minor_classes': 15,
                 'optimizer': 'adam',
                 'learning_rate': 0.00001,
                 'verbose': 2,
                 'dropout_rate': 0.2}

with open("config.ini", 'w') as configfile:
    config.write(configfile)
