
import argparse
import csv
import subprocess
import sys

def _call(arglst):
    try:
        print(arglst)
        subprocess.check_call(arglst, stdout=1, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return e.returncode
    except Exception as e:
        print("subporcess.check_call failed %s" % str(e))
        return 2
    return 0

parser = argparse.ArgumentParser()
parser.add_argument("fname", default="", type=str)
args = parser.parse_args()

fname = args.fname
print("Reading {}".format(fname))
with open(fname, newline="") as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        try:
            assert len(row) == 2, "Should have 2 columns"
            wc = int(row[0])
            ct = float(row[1])
            if 0 != _call(["python", "main.py", "--next_datapoint", str(wc), fname]):
                sys.exit(1)
            if 0 != _call(["python", "main.py", "--finish_datapoint", str(ct), fname]):
                sys.exit(1)
            print("\n")
        except KeyboardInterrupt:
            sys.exit(0)
            

             
