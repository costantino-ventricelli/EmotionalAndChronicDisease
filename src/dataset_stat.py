# coding=utf-8

import sys
sys.path.append("..")

from DatasetManager import HandManager
from collections import Counter
import csv
import os


def main():
    HandManager.set_root_directory()
    gender_list = []
    age_list = []
    state_list = []
    note_list = []
    with open(os.path.join("resource", "complete_diagnosis.csv"), 'r') as file:
        csv_file = csv.reader(file, delimiter=';')
        for row in csv_file:
            gender_list.append(row[0])
            age_list.append(int(row[1]) // 10)
            note_list.append(row[2])
            state_list.append(row[3])
    file.close()
    print(Counter(gender_list).items())
    print(Counter(age_list).items())
    print(Counter(state_list).items())
    print(Counter(note_list).items())


if __name__ == '__main__':
    main()