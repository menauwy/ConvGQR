import csv
import json
import os
from argparse import ArgumentParser
import sys

from tqdm import tqdm

#wiki file in topiocqa dataset folder
# INPUT_FILE = "../../../data1/dataset/topiocqa/full_wiki_segments.tsv"
# output_file = "../../../data1/dataset/topiocqa/bm25_collection/full_wiki_segments_pyserini_format.jsonl"

# # qrecc collection file
# INPUT_FILE = '/home/wangym/data1/dataset/qrecc/new_preprocessed/qrecc_collection.tsv'
# output_file = '/home/wangym/data1/dataset/qrecc/bm25_collection/qrecc_collection_pyserini_format.jsonl'

id_col= 0
text_col= 1
title_col = 2
csv.field_size_limit(sys.maxsize)

def main(dataset):
    # file path
    if dataset == 'qrecc':
        input_file = '/home/wangym/data1/dataset/qrecc/new_preprocessed/qrecc_collection.tsv'
        output_file = '/home/wangym/data1/dataset/qrecc/bm25_collection/qrecc_collection_pyserini_format.jsonl'
        if not os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
    elif dataset == 'topiocqa':
        input_file = "../../../data1/dataset/topiocqa/full_wiki_segments.tsv"
        output_file = "../../../data1/dataset/topiocqa/bm25_collection/full_wiki_segments_pyserini_format.jsonl"
        if not os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # convert to pyserini format
    if dataset == 'qrecc':
        with open(input_file, 'r') as input:
            # reader = csv.reader(input, delimiter="\t")
            reader = csv.reader(x.replace('\0', '') for x in input)
            with open(output_file, 'w') as output:
                for i, row in enumerate(tqdm(reader)):
                    # pid, passsage, no head line
                    if row[id_col] == "id":
                        continue
                    text = row[text_col] if len(row) > 1 else ''
                    # title = row[title_col]
                    obj = {"contents": text, "id": f"doc{i}"}
                    output.write(json.dumps(obj, ensure_ascii=False) + '\n')

    elif dataset == 'topiocqa':
        with open(input_file, 'r') as input:
            reader = csv.reader(input, delimiter="\t")
            with open(output_file, 'w') as output:
                for i, row in enumerate(tqdm(reader)):
                    if row[id_col] == "id":
                        continue
                    title = row[title_col]
                    text = row[text_col]
                    title = ' '.join(title.split(' [SEP] '))
                    obj = {"contents": " ".join([title, text]), "id": f"doc{i}"}
                    output.write(json.dumps(obj, ensure_ascii=False) + '\n')# preserve unicoded characters

if __name__ == "__main__":

    parser = ArgumentParser()
    # parser.add_argument("--wiki_file", type=str, default=INPUT_FILE)
    # parser.add_argument("--output_file", type=str, default=output_file)
    parser.add_argument("--dataset", type=str, default='topiocqa') # qrecc or topiocqa
    args = parser.parse_args()

    # main(wiki_file, output_file, dataset)
    main(args.dataset)

# python convert_to_pyserini_file.py --dataset qrecc