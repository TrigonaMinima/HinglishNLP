from pathlib import Path
from itertools import product


data_dir = Path("../data/transliteration/raw/")


def align_on_words(hin, eng):
    """
    Check if the 2 lists are of equal length.
    """
    return len(eng) == len(hin)


def cross_align(hin, eng):
    """
    Takes a cross product of 2 lists:
    ['1','2'], ['3','4'] will give ['1|3', '1|4', '2|3', '2|4']
    """
    cross_pairs = list(product(eng, hin))
    cross_pairs = ["|".join(i) + "\n" for i in cross_pairs]
    return cross_pairs


true_pairs = []
false_pairs = []
file_name = data_dir / "en_hi_pairs.txt"
with open(file_name, "r") as f:
    for i, pair in enumerate(f):
        print(i, pair)
        eng, hin = pair.strip().split("|")

        eng = eng.replace("-", " ").split(" ")
        hin = hin.replace("-", " ").split(" ")

        flag = align_on_words(hin, eng)
        if flag:
            true_pairs += list(zip(hin, eng))
        else:
            false_pairs += cross_align(hin, eng)


print(len(true_pairs))
with open(data_dir / "en_hi_pairs2.txt", "w") as f:
    for i in true_pairs:
        f.write("|".join(i) + "\n")


print(len(false_pairs))
with open(data_dir / "en_hi_pairs3.txt", "w") as f:
    for i in false_pairs:
        f.write(i)
