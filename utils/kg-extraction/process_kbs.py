import os
import json
import argparse
import pandas as pd
from tqdm import tqdm


def serialize_kb(kb, merge=False):
    out = list()
    for rel in kb["relations"]:
        if rel["head"] == rel["tail"]:
            continue
        out.append(f'{rel["head"]} {rel["type"]} {rel["tail"]}')
    if merge:
        return " ".join(out)
    return out


def main(args):
    rows = list()
    unique_triples = set()

    for fname in tqdm(os.listdir(args.source_dir)):
        fpath = args.source_dir + f"/{fname}"
        with open(fpath, "r") as fp:
            kb = json.load(fp)
            if len(kb["entities"]) == 0:
                continue
            serialized_kb = serialize_kb(kb, args.merge)

            if not args.merge:
                for triple in serialized_kb:
                    if triple in unique_triples:
                        continue
                    rows.append((triple.split(" ")[0], triple))
                    unique_triples.add(triple)
            else:
                if serialized_kb in unique_triples:
                    continue
                rows.append((triple.split(" ")[0], serialized_kb))
                unique_triples.add(serialized_kb)

    df = pd.DataFrame(rows)
    df.to_csv(args.dest_file_name, sep="\t", header=None, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str, required=True)
    parser.add_argument("-o", "--dest_file_name", type=str, required=True)
    parser.add_argument("-m", "--merge", action="store_true")

    args = parser.parse_args()
    main(args)
