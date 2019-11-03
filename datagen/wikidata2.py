import time
import json

from pathlib import Path

from wikidata.wiki_offl import (
    label_pairs, desc_pairs, alias_pairs, get_wikidatajsondump_obj,
    get_all_pairs_entity
)


data_dir = Path("../data/transliteration/")


def write_json(json_string):
    """
    Helper function to check the output of the loaded dict.
    """
    with open("a.json", "w") as f:
        f.write(json_string)


def dump_pairs(filename, pairs):
    """
    Appends the transliteration pairs for each document in the
    supplied filename.
    """
    with open(filename, "a") as f:
        for pair in pairs:
            f.write(pair + "\n")


def process(dump_file, pair_file):
    """
    Reads the wikidata json dump file and writes all the transliteration pairs
    into the pair_file. Also prints the time taken after every 1000 documents
    processed
    """
    t1 = time.time()
    wd = get_wikidatajsondump_obj(dump_file.as_posix())
    for i, entity_dict in enumerate(wd):
        # json_string = json.dumps(entity_dict, indent=4)
        # write_json(json_string)

        pairs = get_all_pairs_entity(entity_dict)
        pairs = set([f"{i}|{j}" for i, j in pairs])

        dump_pairs(pair_file, pairs)

        if i % 1000 == 0:
            t2 = time.time()
            dt = t2 - t1
            print(f"entities: {i} | time: {dt}")
            t1 = t2


if __name__ == "__main__":
    dump_file = data_dir / "latest-all.json.bz2"
    pair_file = data_dir / "en_hi_pairs.txt"
    process(dump_file, pair_file)
    print(f"Written to {pair_file}")
