from pathlib import Path
from difflib import SequenceMatcher

from utils.normalize import strip_accents
from utils.transliterate import transliterations


data_dir = Path("../data/transliteration/raw/")
file_name = data_dir / "en_hi_pairs1.txt"


def check_transliterations(hin, eng):
    """
    Takes in an english and a hindi word. Finds the possible transliterations
    of the hindi word and then checks if the english word is in that list.
    Returns 1 if it is. It also checks if the english word is similar to
    any of the possible transliterations with a score of greater than 0.85.
    Returns 1 if it is. If none of the transliterations are similar to the
    english with with a score of 0.5 then it returns 2.
    """
    t = transliterations(hin)
    if eng in t:
        del t
        return 1

    scores = [SequenceMatcher(None, eng, j).ratio() for j in t] + [0]
    max_score = max(scores)
    if max_score > 0.85:
        del t
        return 1

    if max_score < 0.5:
        del t
        return 2

    return 0


def filter_pair(hin, eng):
    """
    Checks if the english word is a valid transliteration of the hindi word
    based on the possible generated transliterations.
    """
    # transliteration func takes time when length is 11 or more
    if len(hin) > 11:
        return 0

    # observed that majority of the right transliterations had max len
    # differene of 3
    if abs(len(hin) - len(eng)) > 3:
        return 2

    flag = check_transliterations(hin, eng)
    return flag


def filter_pair2(hin, eng):
    """
    Checks if the english word is a valid transliteration of the hindi word
    based on the word endings of english and hindi words.
    """
    endings = {
        "ar": "ड़",
        "ex": "ेक्स",
        "ord": "ोर्ड",
        "ix": "िक्स",
        "ox": "ॉक्स",
        "sor": "जर",
        "sa": "सा",
        # "ine": "इन",
        # "in": "िन",
        "ide": "ाइड",
        "ism": "िज्म",
        "ra": "ड़ा",
        # "and": "ैंड",
        "and": "ैण्ड",
        "phere": "फीयर",
        "xon": "क्सन",
        "ru": "रु",
        "pus": "पस",
        "nge": "ंज",
        "ine": "िन",
        "ine": "ाइन",
        "xar": "सर",
        "ura": "ूरा",
        # "in": "इन"
        "me": "ेम",
        "ery": "री",
        "ite": "ाइट",
        "pur": "पुर",
        "anda": "न्द",
        # "ic": "िक",
        "ist": "िस्ट",
        "ene": "िन",
        "ays": "ेज़"
    }

    for i in endings:
        if eng.endswith(i):
            if hin.endswith(endings[i]):
                return 1
    return 0


with open(file_name, "r") as f:
    for i, pair in enumerate(f):
        # print(i, pair)
        eng, hin = pair.strip().split("|")
        eng = strip_accents(eng)

        flag = filter_pair(hin, eng)
        # flag = filter_pair2(hin, eng)

        if flag == 0:
            with open(data_dir / "false.txt", "a") as f:
                f.write(pair)
        elif flag == 1:
            with open(data_dir / "true.txt", "a") as f:
                f.write(pair)
        elif flag == 2:
            with open(data_dir / "v_false.txt", "a") as f:
                f.write(pair)
