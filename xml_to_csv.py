"""
Usage:
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
"""

import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import json


class_set = set([
    'penny-indian',
    'penny-wheat',
    'penny-lincoln',
    'penny-shield',
    'nickel-liberty',
    'nickel-buffalo',
    'nickel-jefferson',
    'dime-liberty-seated',
    'dime-barber',
    'dime-mercury',
    'dime-roosevelt',
    'quarter-washington',
    'quarter-bicentennial'
])

pcgs_category_map = { 
    'Indian Cent' : 'penny-indian',
    'Lincoln Cent (Wheat Reverse)' : 'penny-wheat',
    'Lincoln Cent (Modern)' : 'penny-lincoln',
    'Liberty Nickel' : 'nickel-liberty',
    'Buffalo Nickel' : 'nickel-buffalo',
    'Jefferson Nickel' : 'nickel-jefferson',
    'Liberty Seated Dime' : 'dime-liberty-seated',
    'Barber Dime' : 'dime-barber',
    'Mercury Dime' : 'dime-mercury',
    'Roosevelt Dime' : 'dime-roosevelt',
    'Washington Quarter' : 'quarter-washington',
}    

pcgs_number_map = {}


def label_text_to_class(row_label):  
     print(row_label.split('-')[0][4:])
     if row_label in class_set:
         return row_label
     elif row_label.split('-')[0][4:] in pcgs_number_map:
         pcgs_record = pcgs_number_map[row_label.split('-')[0][4:]]
         if pcgs_record['category'] == 'Lincoln Cent (Modern)' and 'Shield Reverse' in pcgs_record['sub_category']:
             return 'penny-shield'
         elif pcgs_record['category'] == 'Washington Quarter' and 'Bi-Centennial Reverse' in pcgs_record['sub_category']:
             return 'quarter-bicentennial'
         else:
             return pcgs_category_map[pcgs_record['category']]
     return None


def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines them in a single Pandas datagrame.

    Parameters:
    ----------
    path : {str}
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """
    classes_names = []
    xml_list = []
    
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            label_class = label_text_to_class(member[0].text)
            print(member[0].text)
            print(label_class)
            classes_names.append(label_class)
            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                label_class,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )
            xml_list.append(value)
    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return xml_df, classes_names


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Sample TensorFlow XML-to-CSV converter"
    )
    parser.add_argument(
        "-i",
        "--inputDir",
        help="Path to the folder where the input .xml files are stored",
        type=str,
    )
    parser.add_argument(
        "-o", "--outputFile", help="Name of output .csv file (including path)", type=str
    )

    parser.add_argument(
        "-l",
        "--labelMapDir",
        help="Directory path to save label_map.pbtxt file is specified.",
        type=str,
        default="",
    )

    args = parser.parse_args()

    if args.inputDir is None:
        args.inputDir = os.getcwd()
    if args.outputFile is None:
        args.outputFile = args.inputDir + "/labels.csv"

    assert os.path.isdir(args.inputDir)
    
    print (args.inputDir + '/pcgs_number_map.json')  
    with open(args.inputDir + '/pcgs_number_map.json') as f:
        pcgs_number_map = json.load(f)  
    print(pcgs_number_map)
    
    os.makedirs(os.path.dirname(args.outputFile), exist_ok=True)
    xml_df, classes_names = xml_to_csv(args.inputDir)
    xml_df.to_csv(args.outputFile, index=None)
    print("Successfully converted xml to csv.")
    if args.labelMapDir:
        os.makedirs(args.labelMapDir, exist_ok=True)
        label_map_path = os.path.join(args.labelMapDir, "label_map.pbtxt")
        print("Generate `{}`".format(label_map_path))

        # Create the `label_map.pbtxt` file
        pbtxt_content = ""
        for i, class_name in enumerate(classes_names):
            pbtxt_content = (
                pbtxt_content
                + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(
                    i + 1, class_name
                )
            )
        pbtxt_content = pbtxt_content.strip()
        with open(label_map_path, "w") as f:
            f.write(pbtxt_content)


if __name__ == "__main__":
    main()
