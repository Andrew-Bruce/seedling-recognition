#!/usr/bin/env python
import csv
import os

output_filename = "data_labels.csv"

dir_name = "data/train"
directory = os.fsencode(dir_name)


def iter_all_data_filenames():
    for key, folder in enumerate(os.listdir(directory)):
        sub_folder_name = os.fsdecode(folder)
        print(f"{key}:{sub_folder_name}")
        sub_folder_path = os.path.join(dir_name, sub_folder_name)
        for img_filename in os.listdir(sub_folder_path):
            yield (key, os.path.join(sub_folder_path, img_filename))
    return

def main():
    with open(output_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, dialect='unix')
        csv_writer.writerows(iter_all_data_filenames())
            


main()
