"""
Note that this is based on:
https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

Assumes the working directory has a "dataset" folder which contains a "train" and "test" folder, each containing the
relevant images and their XMLs.
Generates a CSV for train data and one for test data and place inside dataset folder
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # Generate a CSV for train data and one for test data and place inside dataset folder
    for dir_name in ("train", "test"):
        image_path = os.path.join(os.getcwd(), "dataset", dir_name)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(os.path.join("dataset", "{}_labels.csv".format(dir_name)), index=None)
        print('Successfully converted xml to csv.')


main()
