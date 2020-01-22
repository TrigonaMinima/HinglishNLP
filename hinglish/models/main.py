import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from hinglish.models.encoder import Encoder
from hinglish.models.decoder import Decoder
from hinglish.models.attention import Attention
from hinglish.models.s2s import Seq2SeqInfer as Seq2Seq
from hinglish.utils.modelio import load_torch_pickle


def build_s2s_model(model_conf):
    device = torch.device(model_conf["device"])

    src_pk_path = model_conf["src_pk_path"]
    SRC = load_torch_pickle(src_pk_path)

    trg_pk_path = model_conf["trg_pk_path"]
    TRG = load_torch_pickle(trg_pk_path)

    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)

    enc_emb_dim = 256
    dec_emb_dim = 256

    enc_hid_dim = 512
    dec_hid_dim = 512

    enc_dropout = 0.5
    dec_dropout = 0.5

    pad_idx = SRC.vocab.stoi['<pad>']
    sos_idx = TRG.vocab.stoi['<s>']
    eos_idx = TRG.vocab.stoi['<e>']

    enc = Encoder(
        input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
    attn = Attention(enc_hid_dim, dec_hid_dim)
    dec = Decoder(
        output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)

    model = Seq2Seq(enc, dec, pad_idx, sos_idx, eos_idx, device).to(device)

    model_path = model_conf["model_path"]
    model.load_state_dict(torch.load(model_path))

    model_dict = {
        "src": SRC,
        "trg": TRG,
        "model": model,
        "device": device
    }
    return model_dict
