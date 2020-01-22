import math

import torch

from hinglish.utils.text import numericalize_src, unnumericalize_trg


def normalize(model_dict, word, topk=1):
    SRC = model_dict["src"]
    TRG = model_dict["trg"]
    model = model_dict["model"]
    device = model_dict["device"]

    model.eval()

    numericalized = numericalize_src(SRC, word)
    in_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)

    sentence_length = torch.LongTensor([len(numericalized)]).to(device)

    outs = model(in_tensor, sentence_length, beam=topk)

    words = []
    for out, score in outs:
        fin_out = unnumericalize_trg(TRG, out)
        res = {
            "orig": word,
            "norm": fin_out,
            "score": math.exp(score)
        }
        words.append(res)
    return words
