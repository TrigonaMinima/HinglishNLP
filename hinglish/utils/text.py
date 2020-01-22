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


def tokenize(string):
    """
    Tokenizes a string into characters. Returns the tokens in reverse
    order as it might make optimization problem easier. Will read paper
    to understand this.
    """
    return list(string)


def numericalize_src(SRC, word):
    tokenized = tokenize(word)
    tokenized = ['<s>'] + [t.lower() for t in tokenized] + ['<e>']
    numericalized = [SRC.vocab.stoi[t] for t in tokenized]
    return numericalized


def unnumericalize_trg(TRG, out):
    fin_out = []
    for t in out[1:]:
        # print(t)
        if TRG.vocab.itos[t] == "<e>":
            break

        if TRG.vocab.itos[t] == "<pad>":
            fin_out.append(" ")
        elif TRG.vocab.itos[t] == "<unk>":
            fin_out.append("")
        else:
            fin_out.append(TRG.vocab.itos[t])

    return "".join(fin_out)
