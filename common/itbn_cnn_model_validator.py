# IMPORTS ##############################################
import os
import networkx as nx
from datetime import datetime

# cnn models
from opt_classifier import opt_classifier
from aud_classifier import aud_classifier

# file io
from common.itbn_pipeline import *

# itbn model
from pgmpy.models import IntervalTemporalBayesianNetwork

# PARAMETERS ##############################################

# boolean used to switch between training/validation
validation = False

# itbn model path
ITBN_MODEL_PATH = 'input/itbn.nx'

# cnn models paths
AUD_DQN_CHKPNT = "../aud_classifier/itbn_aud_final/model.ckpt"
OPT_DQN_CHKPNT = "../opt_classifier/itbn_opt_final/model.ckpt"

# tf records path
TF_RECORDS_PATH = "../../ITBN_tfrecords/"
LABELS_PATH = '/home/assistive-robotics/PycharmProjects/dbn_arl/labels/'

# cnn parameters
ALPHA = 1e-5
AUD_FRAME_SIZE = 20
AUD_STRIDE = 7
OPT_FRAME_SIZE = 45
OPT_STRIDE = 20

# debug characters for cnn classifications [silence, robot, human]
SEQUENCE_CHARS = ["_", "|", "*"]

# subset of allen's interval relations that are used to classify a window fed to the cnns
INTERVAL_RELATION_MAP = {
    (1., -1., -1., 1.): 'DURING',
    (-1., 1., -1., 1.): 'DURING_INV',
    (-1., -1., -1., 1.): 'OVERLAPS',
    (1., 1., -1., 1.): 'OVERLAPS_INV',
    (0., -1., -1., 1.): 'STARTS',
    (0., 1., -1., 1.): 'STARTS_INV',
    (1., 0., -1., 1.): 'FINISHES',
    (-1., 0., -1., 1.): 'FINISHES_INV',
    (0., 0., -1., 1.): 'EQUAL'
}


# FUNCTIONS DEFINITION ##############################################

# given a window and event start and end time calculates the interval relation between them
def calculate_relationship(window_s, window_e, event_s, event_e):
    temp_distance = (np.sign(event_s - window_s), np.sign(event_e - window_e),
                     np.sign(event_s - window_e), np.sign(event_e - window_s))
    return INTERVAL_RELATION_MAP.get(temp_distance, '')


# verifies if there is a valid interval relation between the event and the given window
def overlaps(s_time, e_time, td, event_name):
    s_label = event_name + "_s"
    e_label = event_name + "_e"

    if s_label in td and calculate_relationship(s_time, e_time, td[s_label], td[e_label]) != '':
        return True
    return False


# determines the predicted labels for the audio data
def label_data_aud(frame_size, stride, frame_num, sequence_len, td):
    predicted_label_data = np.zeros((BATCH_SIZE, AUD_CLASSES)).astype(float)
    aud_label = 0

    s_frame = stride * frame_num
    e_frame = s_frame + frame_size

    if e_frame > sequence_len:
        e_frame = sequence_len

    if overlaps(s_frame, e_frame, td, "command"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "prompt"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "reward"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "abort"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "noise_0"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "noise_1"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "audio_0"):
        aud_label = 2
    if overlaps(s_frame, e_frame, td, "audio_1"):
        aud_label = 2

    predicted_label_data[0][aud_label] = 1
    return predicted_label_data


# determines the predicted labels for the video data
def label_data_opt(frame_size, stride, frame_num, sequence_len, td):
    predicted_label_data = np.zeros((BATCH_SIZE, OPT_CLASSES)).astype(float)
    opt_label = 0

    s_frame = stride * frame_num
    e_frame = s_frame + frame_size

    if e_frame > sequence_len:
        e_frame = sequence_len

    if overlaps(s_frame, e_frame, td, "command"):
        opt_label = 1
    if overlaps(s_frame, e_frame, td, "prompt"):
        opt_label = 1
    if overlaps(s_frame, e_frame, td, "noise_0"):
        opt_label = 1
    if overlaps(s_frame, e_frame, td, "noise_1"):
        opt_label = 1
    if overlaps(s_frame, e_frame, td, "gesture_0"):
        opt_label = 2
    if overlaps(s_frame, e_frame, td, "gesture_1"):
        opt_label = 2

    predicted_label_data[0][opt_label] = 1
    return predicted_label_data


if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    # MODELS PREPARATION ##############################################
    # read contents of TFRecord file and generate list of file names
    file_names = list()
    for root, directory, files in os.walk(TF_RECORDS_PATH):
        for f in files:
            if validation:
                if 'validation' in f:
                    file_names.append(os.path.join(root, f))
            else:
                if 'validation' not in f:
                    file_names.append(os.path.join(root, f))
    file_names.sort()
    print("{}".format(file_names))

    # loading itbn model
    nx_model = nx.read_gpickle(ITBN_MODEL_PATH)
    model = IntervalTemporalBayesianNetwork(nx_model.edges())
    model.add_cpds(*nx_model.cpds)
    model.learn_temporal_relationships_from_cpds()

    # load cnn models
    aud_dqn = aud_classifier.ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA,
                                             filename=AUD_DQN_CHKPNT)
    opt_dqn = opt_classifier.ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA,
                                             filename=OPT_DQN_CHKPNT)

    # prepare tf objects
    aud_coord = tf.train.Coordinator()
    opt_coord = tf.train.Coordinator()

    # read records from files into tensors
    seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, file_name = \
        input_pipeline(file_names)

    # initialize variables
    with aud_dqn.sess.as_default():
        with aud_dqn.graph.as_default():
            aud_dqn.sess.run(tf.local_variables_initializer())
            aud_dqn.sess.graph.finalize()
            threads = tf.train.start_queue_runners(coord=aud_coord, sess=aud_dqn.sess)

    with opt_dqn.sess.as_default():
        with opt_dqn.graph.as_default():
            opt_dqn.sess.run(tf.local_variables_initializer())
            opt_dqn.sess.graph.finalize()
            threads = tf.train.start_queue_runners(coord=opt_coord, sess=opt_dqn.sess)

    print("Num epochs: {}\nBatch Size: {}\nNum Files: {}".format(NUM_EPOCHS, BATCH_SIZE,
                                                                 len(file_names)))

    # confusion matrices
    aud_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    opt_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # debugging sequences
    aud_sequences = dict()
    opt_sequences = dict()

    num_files = len(file_names)
    counter = 0
    while len(file_names) > 0:
        # read a batch of tfrecords into np arrays
        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = opt_dqn.sess.run(
         [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, file_name])

        if validation:
            name = name[0].replace('.txt', '_validation.tfrecord').replace(LABELS_PATH,
                                                                           TF_RECORDS_PATH)
        else:
            name = name[0].replace('.txt', '.tfrecord').replace(LABELS_PATH, TF_RECORDS_PATH)

        if name in file_names:
            counter += 1
            print("processing {}/{}: {}".format(counter, num_files, name))
            file_names.remove(name)
            timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])
            aud_num_chunks = (seq_len - AUD_FRAME_SIZE) / AUD_STRIDE + 1
            opt_num_chunks = (seq_len - OPT_FRAME_SIZE) / OPT_STRIDE + 1
            aud_chunk_counter = 0
            opt_chunk_counter = 0
            aud_frame_counter = 0
            opt_frame_counter = 0
            aud_real_sequence = ""
            aud_pred_sequence = ""
            opt_real_sequence = ""
            opt_pred_sequence = ""
            process_aud_window = False
            process_opt_window = False

            while aud_chunk_counter < aud_num_chunks or opt_chunk_counter < opt_num_chunks:
                aud_frame_counter += 1
                opt_frame_counter += 1

                if aud_frame_counter == AUD_FRAME_SIZE and aud_chunk_counter < aud_num_chunks:
                    process_aud_window = True
                    aud_frame_counter = 0
                if opt_frame_counter == OPT_FRAME_SIZE and opt_chunk_counter < opt_num_chunks:
                    process_opt_window = True
                    opt_frame_counter = 0

                if process_aud_window:
                    with aud_dqn.sess.as_default():
                        aud_label_data = label_data_aud(AUD_FRAME_SIZE, AUD_STRIDE,
                                                        aud_chunk_counter, seq_len, timing_dict)
                        vals = {
                            aud_dqn.seq_length_ph: seq_len,
                            aud_dqn.aud_ph: np.expand_dims(
                                aud_raw[0][AUD_STRIDE * aud_chunk_counter:
                                           AUD_STRIDE * aud_chunk_counter + AUD_FRAME_SIZE], 0),
                            aud_dqn.aud_y_ph: aud_label_data
                        }
                        aud_pred = aud_dqn.sess.run([aud_dqn.aud_observed], feed_dict=vals)
                        real_class = int(np.argmax(aud_label_data))
                        selected_class = int(aud_pred[0][0])
                        aud_matrix[real_class][selected_class] += 1
                        aud_real_sequence += SEQUENCE_CHARS[real_class]
                        aud_pred_sequence += SEQUENCE_CHARS[selected_class]
                        aud_chunk_counter += 1
                        process_aud_window = False
                if process_opt_window:
                    with opt_dqn.sess.as_default():
                        opt_label_data = label_data_opt(OPT_FRAME_SIZE, OPT_STRIDE,
                                                        opt_chunk_counter, seq_len, timing_dict)
                        vals = {
                            opt_dqn.seq_length_ph: seq_len,
                            opt_dqn.pnt_ph: np.expand_dims(
                                opt_raw[0][OPT_STRIDE * opt_chunk_counter:
                                           OPT_STRIDE * opt_chunk_counter + OPT_FRAME_SIZE], 0),
                            opt_dqn.pnt_y_ph: opt_label_data
                        }
                        opt_pred = opt_dqn.sess.run([opt_dqn.wave_observed], feed_dict=vals)
                        real_class = int(np.argmax(opt_label_data))
                        selected_class = int(opt_pred[0][0])
                        opt_matrix[real_class][selected_class] += 1
                        opt_real_sequence += SEQUENCE_CHARS[real_class]
                        opt_pred_sequence += SEQUENCE_CHARS[selected_class]
                        opt_chunk_counter += 1
                        process_opt_window = False
            aud_sequences[name] = aud_real_sequence + "\n" + aud_pred_sequence
            opt_sequences[name] = opt_real_sequence + "\n" + opt_pred_sequence

    # print results
    print("time end: {}\nAUDIO\n{}\n\nVIDEO\n{}\n".format(datetime.now(), aud_matrix, opt_matrix))

    print("\n\nAUDIO SEQUENCES:")
    for f in aud_sequences.keys():
        print("{}\n{}\n".format(f, aud_sequences[f]))

    print("\n\nVIDEO SEQUENCES:")
    for f in opt_sequences.keys():
        print("{}\n{}\n".format(f, opt_sequences[f]))
