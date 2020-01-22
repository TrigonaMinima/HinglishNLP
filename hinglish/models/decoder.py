import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, trg_vocab, dec_emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.vocab_size = trg_vocab

        self.attn = attention

        self.emb = nn.Embedding(trg_vocab, dec_emb_dim)
        self.dropout = nn.Dropout(dropout)

        gru_in_dim = dec_emb_dim + (enc_hid_dim * 2)
        self.gru = nn.GRU(gru_in_dim, dec_hid_dim)

        lin_in_dim = dec_emb_dim + (enc_hid_dim * 2) + dec_hid_dim
        self.linear = nn.Linear(lin_in_dim, trg_vocab)

    def forward(self, trg, hid, enc_outs, mask):
        trg = trg.unsqueeze(0)
        trg = self.emb(trg)
        trg = self.dropout(trg)

        a = self.attn(enc_outs, hid, mask)
        a = a.unsqueeze(1)

        enc_outs = enc_outs.permute(1, 0, 2)
        weighted = torch.bmm(a, enc_outs)

        weighted = weighted.permute(1, 0, 2)
        gru_input = torch.cat((trg, weighted), dim=2)
        out, hid = self.gru(gru_input, hid.unsqueeze(0))

        assert (out == hid).all()

        trg = trg.squeeze(0)
        out = out.squeeze(0)
        weighted = weighted.squeeze(0)

        out = self.linear(torch.cat((out, weighted, trg), dim=1))
        return out, hid.squeeze(0), a.squeeze(1)
