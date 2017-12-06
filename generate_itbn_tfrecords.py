#!/usr/bin/env python

# generate_itbn_tfrecords.py
# Madison Clark-Turner
# 12/2/2017
import tensorflow as tf
import numpy as np
from dqn_packager_itbn import *
import rosbag
import rospy
import heapq

from constants import *

import os
from os.path import isfile, join

from itbn_tfrecord_rw import *

topic_names = [
    '/action_finished',
    '/nao_robot/camera/top/camera/image_raw',
    '/nao_robot/microphone/naoqi_microphone/audio_raw'
]


def readTimingFile(filename):
    # generate a heap of timing event tuples
    ifile = open(filename, 'r')

    timing_queue = []

    line = ifile.readline()
    while (len(line) != 0):
        line = line.split()
        # tuple = (time of event, name of event)
        event_time = float(line[1])
        event_time = rospy.Duration(event_time)
        timing_queue.append((event_time, line[0]))
        line = ifile.readline()

    heapq.heapify(timing_queue)

    return timing_queue


def gen_TFRecord_from_file(out_dir, out_filename, bag_filename, timing_filename, flip=False):
    packager = DQNPackager(flip=flip)
    bag = rosbag.Bag(bagfile)

    packager.p = False

    #######################
    ##   TIMING FILE    ##
    #######################

    # parse timing file
    timing_queue = readTimingFile(timing_filename)
    # get first timing event
    current_time = heapq.heappop(timing_queue)
    timing_dict = {}

    #######################
    ##     READ FILE     ##
    #######################

    all_timing_frames_found = False
    start_time = None

    for topic, msg, t in bag.read_messages(topics=topic_names):
        if (start_time == None):
            start_time = t

        if (not all_timing_frames_found and t > start_time + current_time[0]):
            # add the frame number anf timing label to frame dict
            timing_dict[current_time[1]] = packager.getFrameCount()
            if (len(timing_queue) > 0):
                current_time = heapq.heappop(timing_queue)
            else:
                all_timing_frames_found = True

        elif (topic == topic_names[1]):
            packager.imgCallback(msg)
        elif (topic == topic_names[2]):
            packager.audCallback(msg)

    # perform data pre-processing steps
    packager.formatOutput()

    print(timing_dict)

    # generate TFRecord data
    ex = make_sequence_example(
        packager.getImgStack(), img_dtype,
        packager.getPntStack(), pnt_dtype,
        packager.getAudStack(), aud_dtype,
        timing_dict,
        timing_filename)

    # write TFRecord data to file
    end_file = ".tfrecord"
    if (flip):
        end_file = "_flip" + end_file

    writer = tf.python_io.TFRecordWriter(out_dir + out_filename + end_file)
    writer.write(ex.SerializeToString())
    writer.close()

    packager.reset()
    bag.close()


if __name__ == '__main__':
    gen_single_file = True
    view_single_file = True
    process_all_files = False

    rospy.init_node('gen_tfrecord', anonymous=True)

    #############################

    bagfile = os.environ["HOME"] + "/ITBN_bags/test_03/zga1.bag"
    timefile = os.environ["HOME"] + "/PycharmProjects/dbn_arl/labels/test_03/zga1.txt"

    outfile = "/tfrecords/scrap.tfrecord"
    outdir = os.environ["HOME"] + "/ITBN_tfrecords/"

    #############################

    if (gen_single_file):
        # generate a single file and store it as a scrap.tfrecord; Used for Debugging

        gen_TFRecord_from_file(out_dir=outdir, out_filename="scrap", bag_filename=bagfile,
                               timing_filename=timefile, flip=False)

    #############################

    if (view_single_file):
        # read contents of scrap.tfrecord; Used for Debugging

        print("READING...")
        coord = tf.train.Coordinator()
        filename_queue = tf.train.string_input_producer([outfile])

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            # parse TFrecord
            context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
            threads = tf.train.start_queue_runners(coord=coord)


            def processData(inp, data_type):
                data_s = tf.reshape(inp, [-1, data_type["cmp_h"], data_type["cmp_w"],
                                          data_type["num_c"]])
                return tf.cast(data_s, tf.uint8)


            # information to extract from TFrecord
            seq_len = context_parsed["length"]
            # img_raw = processData(sequence_parsed["img_raw"], img_dtype)
            opt_raw = processData(sequence_parsed["opt_raw"], pnt_dtype)
            aud_raw = processData(sequence_parsed["aud_raw"], aud_dtype)
            timing_labels = context_parsed["timing_labels"]
            timing_values = sequence_parsed["timing_values"]
            name = context_parsed["example_id"]

            # extract data (set range to a val > 1 if multiple TFrecords stored in single file)
            for i in range(1):
                # l, i, p, a, tl, tv, n = sess.run([seq_len, img_raw, opt_raw, aud_raw, timing_labels, timing_values, name]) #<-- has img data
                l, p, a, tl, tv, n = sess.run(
                    [seq_len, opt_raw, aud_raw, timing_labels, timing_values, name])
                timing_dict = parse_timing_dict(tl, tv)

                print(name)
                print(timing_dict)

            coord.request_stop()
            coord.join(threads)


            # Use for visualizing Data Types
            def show(data, d_type):
                tout = []
                out = []
                for i in range(data.shape[0]):
                    imf = np.reshape(data[i], (d_type["cmp_h"], d_type["cmp_w"], d_type["num_c"]))

                    limit_size = 64

                    if (d_type["cmp_h"] > limit_size):
                        mod = limit_size / float(d_type["cmp_h"])
                        imf = cv2.resize(imf, None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)
                    if (imf.shape[2] == 2):
                        imf = np.concatenate((imf, np.zeros((d_type["cmp_h"], d_type["cmp_w"], 1))),
                                             axis=2)
                        imf[..., 0] = imf[..., 1]
                        imf[..., 2] = imf[..., 1]
                        imf = imf.astype(np.uint8)

                    if (i % 10 == 0 and i != 0):
                        if (len(tout) == 0):
                            tout = out.copy()
                        else:
                            tout = np.concatenate((tout, out), axis=0)

                        out = []
                    if (len(out) == 0):
                        out = imf
                    else:
                        out = np.concatenate((out, imf), axis=1)

                if (data.shape[0] % 10 != 0):
                    fill = np.zeros((limit_size, limit_size * (10 - (data.shape[0] % 10)),
                                     d_type["num_c"]))  # .fill(255)
                    fill.fill(255)
                    out = np.concatenate((out, fill), axis=1)

                return tout


            print("file length: ", l)

            show_from = 110
            img = show(p[show_from:], pnt_dtype)
            # img2 = show(i[show_from:], img_dtype)
            cv2.imshow("img", img)
            # cv2.imshow("pnt", img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        #############################

        # states = ["compliant", "noncompliant"]

    if (process_all_files):
        path = os.environ[
                   "HOME"] + '/' + "Documents/AssistiveRobotics/AutismAssistant/pomdpData/long_sess_0"
        outdir = os.environ["HOME"] + '/' + "catkin_ws/src/deep_q_network/tfrecords/long/"
        for i in range(1, 2):
            file_dir = path + str(i)
            filename_list = [file_dir + '/' + f for f in os.listdir(file_dir) if
                             isfile(join(file_dir, f))]
            filename_list.sort()
            for f in filename_list:
                state = 1
                if (f.find("none") >= 0):
                    state = 0

                # tag = f[f.find("test") + len("test_00")+1:-(len(".bag"))]
                tag = f
                while (tag.find("/") >= 0):
                    tag = tag[tag.find("/") + 1:]
                tag = tag[:-(len(".bag"))]
                new_name = "long_sees_0" + str(i) + '_' + tag
                print(tag + "......." + new_name)
                index = i
                # print("truth tests:", f.find("1.bag"), i >=3)
                # if(f.find("1.bag") >= 0 and i >=3):
                index = -1

                # writer = tf.python_io.TFRecordWriter(outdir+new_name + ".tfrecord")
                gen_TFRecord_from_file(outdir=outdir, bagfile=f, state=state, name=new_name,
                                       flip=False, index=index)
            # writer.close()

            # writer = tf.python_io.TFRecordWriter(outdir+new_name + "_flip.tfrecord")
            # gen_TFRecord_from_file(outdir=outdir, bagfile=f, state=state, name=new_name, flip=True, index=index)
            # writer.close()
