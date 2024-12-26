import os
import csv
import json
from tqdm import tqdm
import argparse


def preprocess_corpus(data_dir, output_dir):
    corpus_path = os.path.join(data_dir, "wiki_webq_corpus.tsv")
    output_path = os.path.join(output_dir, "webq-corpus.jsonl")

    with open(corpus_path, "r", encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        data = list(reader)

    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        for row in tqdm(data):
            json_line = {
                "docid": row["id"],  # 将 "id" 映射为 "docid"
                "text": row["text"],  # 将 "text" 映射为 "text"
                "title": row["title"],  # 将 "title" 映射为 "title"
            }
            jsonl_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Corpus file has been saved to {output_path}.")


def convert_train_or_dev(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = json.load(f)
    idx = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for line in tqdm(lines):
            if len(line["positive_ctxs"]) < 1 or len(line["hard_negative_ctxs"]) < 1:
                continue

            item = {
                "query_id": str(idx),
                "query": line["question"],
                "answers": line["answers"],
                "positive_passages": [],
                "negative_passages": [],
            }

            for pos_psg in line["positive_ctxs"]:
                curr_psg = {
                    "docid": (
                        pos_psg.pop("psg_id")
                        if "psg_id" in pos_psg
                        else pos_psg.pop("passage_id")
                    ),  # passage_id, psg_id
                    "title": pos_psg.pop("title"),
                    "text": pos_psg.pop("text"),
                }
                item["positive_passages"].append(curr_psg)

            for neg_psg in line["hard_negative_ctxs"]:
                curr_psg = {
                    "docid": (
                        neg_psg.pop("psg_id")
                        if "psg_id" in neg_psg
                        else neg_psg.pop("passage_id")
                    ),
                    "title": neg_psg.pop("title"),
                    "text": neg_psg.pop("text"),
                }
                item["negative_passages"].append(curr_psg)

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            idx += 1


def convert_test(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="\t")
        data = list(reader)

    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        query_id = 0
        for row in tqdm(data):
            item = {
                "query_id": str(query_id),
                "query": row[0],  # 第一列作为 query
                "answers": eval(row[1]),  # 第二列作为 answers
            }
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            query_id += 1


def preprocess_dataset(data_dir, output_dir):
    train_path = os.path.join(data_dir, "webq-train.json")
    output_train_path = os.path.join(output_dir, "webq-train.jsonl")
    convert_train_or_dev(train_path, output_train_path)
    print(f"Train file has been saved to {output_train_path}.")

    dev_path = os.path.join(data_dir, "webq-dev.json")
    output_dev_path = os.path.join(output_dir, "webq-dev.jsonl")
    convert_train_or_dev(dev_path, output_dev_path)
    print(f"Dev file has been saved to {output_dev_path}.")

    test_path = os.path.join(data_dir, "webq-test.csv")
    output_test_path = os.path.join(output_dir, "webq-test.jsonl")
    convert_test(test_path, output_test_path)
    print(f"Test file has been saved to {output_test_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, metavar="path", help="Path to original data file."
    )
    parser.add_argument(
        "--output_dir", type=str, metavar="path", help="Path to output file."
    )
    args = parser.parse_args()

    preprocess_corpus(args.data_dir, args.output_dir)
    preprocess_dataset(args.data_dir, args.output_dir)
