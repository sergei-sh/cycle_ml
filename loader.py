"""Takes an input csv in the following format:
WAFER_COUNT, CYCLE_TIME
and sends it to main.py according to usual protocol (see README),
uses input filename as tool_recipe id;

Has some (disabled) abitlity to log results to tensorboards
""" 
import argparse
import csv
import subprocess
import sys

import tensorflow as tf

from cycle_ml.aux import my_call

parser = argparse.ArgumentParser()
parser.add_argument("fname", default="", type=str)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

with tf.Session() as sess:
    err = tf.Variable(6)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    writer = tf.summary.FileWriter("tboard", sess.graph)
    total_error = tf.placeholder(tf.float32, name="error")
    tf.summary.scalar("error", total_error)
    summaries = tf.summary.merge_all()

    fname = args.fname
    print("Reading {}".format(fname))
    with open(fname, newline="") as csv_file:
        reader = csv.reader(csv_file)
        step = 0
        for row in reader:
            try:
                assert len(row) == 2, "Should have 2 columns"
                wc = float(row[0])
                ct = float(row[1])
                print("\n")
                cmd = ["python", "main.py", "--next_datapoint", str(wc), fname]
                if args.verbose:
                    cmd.append("--verbose")
                if 0 != my_call(cmd):
                    sys.exit(1)
                print("\n")
                abs_err = my_call(["python", "main.py", "--finish_datapoint", str(ct), fname])

                print("ERROR ", abs_err)
                feed_dict = { total_error: abs_err }
                [summary] = sess.run([summaries], feed_dict)
                writer.add_summary(summary, step)
                step += 1

                #print("\n")
            except KeyboardInterrupt:
                sys.exit(0)
                

             
