import unicodedata


def strip_accents(text):
    """
    - Normalizes the accented characters
    - Truncates the foreign characters
    """
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore")
    text = text.decode("utf8")
    return text
