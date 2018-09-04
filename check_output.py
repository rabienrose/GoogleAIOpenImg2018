import argparse
import json

hierarchy_file = '/home/chamo/Documents/work/OpenImgChamo/config/bbox_labels_500_hierarchy.json'
result_file = '/home/chamo/Documents/data/UntitledFolder/expanded_test.csv'

record_count=0
with open(result_file, "r") as f_result:
    line = f_result.readline()
    total_add_count=0
    while True:
        line = f_result.readline()
        if line == '':
            break
        record_count = record_count + 1
        if record_count==42545:
            print(line)
        splited = line.split(",")
        if len(splited)>2:
            print(splited)
    print(record_count)

