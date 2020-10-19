import sys
import os
import json
import numpy as np

import utils

class TCGDB(object):

    def __init__(self, root):

        self.root = root
        self.name = "tcg"

        self.path = os.path.join(root, "dataset", self.name)

        self.db_file = os.path.join(self.path, "tcg.json")
        self.data_file = os.path.join(self.path, "tcg_data.npy")

        # database
        self.dataset = ""
        self.description = ""
        self.joint_dictionary = {}
        self.version = []
        self.sequences = []

        # learning protocols
        self.sampling_factor = 5  # to subsample 100 Hz to 20 Hz
        self.train_subclasses = False
        self.train_test_sets = {"xs": [[[1, 2, 3, 4], [5]],
                                       [[1, 3, 4, 5], [2]],
                                       [[1, 2, 4, 5], [3]],
                                       [[1, 2, 3, 5], [4]],
                                       [[2, 3, 4, 5], [1]]],
                                "xv": [[["right", "bottom", "left"], ["top"]],
                                       [["right", "bottom", "top"], ["left"]],
                                       [["bottom", "left", "top"], ["right"]],
                                       [["right", "left", "top"], ["bottom"]]]}

    def set_subclasses(self, train_subclasses=0):

        if train_subclasses == 0:
            self.train_subclasses = False
        elif train_subclasses == 1:
            self.train_subclasses = True
        else:
            sys.exit("Subclasses parameter has to be boolean (0/1), got {}. Abort!".format(train_subclasses))

    def get_nb_classes(self):

        if self.train_subclasses:
            return 15
        else:
            return 4

    def get_number_runs(self, protocol_type="xs"):

        return len(self.train_test_sets[protocol_type])

    def get_train_test_set(self, run_id=1, protocol_type="xs"):

        return self.train_test_sets[protocol_type][run_id]

    def get_train_test_data(self, run_id=1, protocol_type="xs"):

        # initialize train and test
        X_train = []
        Y_train = []

        X_test = []
        Y_test = []

        # sets
        train_sets = self.train_test_sets[protocol_type][run_id][0]
        test_sets = self.train_test_sets[protocol_type][run_id][1]


        # iterate through sequences
        for _, seq in enumerate(self.sequences):

            # poses & labels
            poses = np.array([f.pose.flatten() for f in seq.frames])

            if self.train_subclasses:
                targets = np.array([f.min_cls for f in seq.frames])
            else:
                targets = np.array([f.maj_cls for f in seq.frames])

            targets_one_hot = utils.one_hot_encoding(targets, nb_classes=self.get_nb_classes())

            # subsampling
            poses = utils.subsampling(poses, sampling_factor=self.sampling_factor)
            targets_one_hot = utils.subsampling(targets_one_hot, sampling_factor=self.sampling_factor)

            # assign to set
            if protocol_type == "xs":

                if seq.subject in train_sets:

                    X_train.append(poses)
                    Y_train.append(targets_one_hot)

                elif seq.subject in test_sets:

                    X_test.append(poses)
                    Y_test.append(targets_one_hot)

                else:

                    print("Sequence is not contained in TRAIN neither in TEST...")

            elif protocol_type == "xv":

                if seq.viewpoint in train_sets:

                    X_train.append(poses)
                    Y_train.append(targets_one_hot)

                elif seq.viewpoint in test_sets:

                    X_test.append(poses)
                    Y_test.append(targets_one_hot)

                else:

                    print("Sequence is not contained in TRAIN neither in TEST...")

        # maximal sequence length
        max_seq_len_train = max([len(s) for s in X_train])
        max_seq_len_test = max([len(s) for s in X_test])
        max_seq_len = max([max_seq_len_train, max_seq_len_test])

        # zero padding
        X_train = np.array([utils.pad_sequence(s, max_seq_len) for s in X_train])
        Y_train = np.array([utils.pad_sequence(s, max_seq_len) for s in Y_train])
        X_test = np.array([utils.pad_sequence(s, max_seq_len) for s in X_test])
        Y_test = np.array([utils.pad_sequence(s, max_seq_len) for s in Y_test])

        return X_train, Y_train, X_test, Y_test

    def open(self):

        # load
        with open(self.db_file, "rb") as fin:
            db = json.load(fin)

        with open(self.data_file, "rb") as fin:
            data = np.load(fin, allow_pickle=True)

        # meta data
        self.dataset = db["dataset"]
        self.description = db["description"]
        self.version = db["version"]
        self.joint_dictionary = db["joint_dictionary"]

        # sequences
        for sid, seq in enumerate(db["sequences"]):

            sequence = TCGSequence()

            # sequence description
            sequence.subject = seq["subject_id"]
            sequence.junction = seq["junction"]
            sequence.scene = seq["scene_id"]

            sequence.scene_agents = seq["scene_agents"]

            agent_description = sequence.scene_agents[seq["agent_number"]]
            sequence.id = agent_description["id"]
            sequence.viewpoint = agent_description["position"]
            sequence.intention = agent_description["intention"]
            sequence.queue = agent_description["queue"]

            sequence.annotation = seq["annotation"]

            sequence.num_frames = seq["num_frames"]
            sequence.frames = []

            # annotations
            maj_class_name = [None] * sequence.num_frames
            min_class_name = [None] * sequence.num_frames

            for li, label_interval in enumerate(sequence.annotation):
                for f in range(label_interval[2], label_interval[3]):
                    maj_class_name[f] = label_interval[0]
                    min_class_name[f] = label_interval[1]

            # create frame instance
            pose_sequence = data[sid]
            for p, pose in enumerate(pose_sequence):
                frame = TCGFrame()
                frame.pose = pose
                frame.maj_cls_name = maj_class_name[p]
                frame.min_cls_name = min_class_name[p]
                frame.encode_majlabel()
                frame.encode_minlabel()
                sequence.frames.append(frame)

            # append
            self.sequences.append(sequence)


class TCGSequence(object):

    def __init__(self):

        self.subject = int()

        self.junction = ""
        self.scene = int()
        self.id = int()
        self.viewpoint = int()
        self.intention = int()
        self.queue = int()

        self.annotation = []

        self.num_frames = int()
        self.frames = []


class TCGFrame(object):

    def __init__(self):
        self.pose = []
        self.maj_cls_name = ""
        self.maj_cls = -1
        self.min_cls_name = ""
        self.min_cls = -1

    def encode_majlabel(self):

        maj_label_dict = {"inactive": 0, "stop": 1, "go": 2, "clear": 3}

        self.maj_cls = maj_label_dict[self.maj_cls_name]

    def encode_minlabel(self):

        min_label_dict = {"inactive_normal-pose": 0, "inactive_out-of-vocabulary": 0, "inactive_transition": 0,
                          "stop_both-static": 1, "stop_both-dynamic": 2, "stop_left-static": 3,
                          "stop_left-dynamic": 4, "stop_right-static": 5, "stop_right-dynamic": 6,
                          "clear_left-static": 7, "clear_right-static": 8, "go_both-static": 9,
                          "go_both-dynamic": 10, "go_left-static": 11, "go_left-dynamic": 12,
                          "go_right-static": 13, "go_right-dynamic": 14}

        self.min_cls = min_label_dict[self.maj_cls_name + "_" + self.min_cls_name]


if __name__ == "__main__":

    protocol_type = "xv"
    train_subclasses = False

    # load db
    db = TCGDB(os.getcwd())

    db.open()

    print("Finished opening TCG dataset")

    # load train and test sets
    db.set_subclasses(train_subclasses=train_subclasses)

    for run_id in range(db.get_number_runs(protocol_type=protocol_type)):
        X_train, Y_train, X_test, Y_test = db.get_train_test_data(run_id=run_id, protocol_type=protocol_type)

        print("\n{:10}: ".format("Run"), db.get_train_test_set(run_id=run_id, protocol_type=protocol_type))
        print("{:10}: ".format("X_train"), X_train.shape, " | {:10}: ".format("Y_train"), Y_train.shape)
        print("{:10}: ".format("X_test"), X_test.shape, " | {:10}: ".format("Y_test"), Y_test.shape)

