import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, inp_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.inp_dim = inp_dim
        self.enc_emb_dim = enc_emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.emb = nn.Embedding(inp_dim, enc_emb_dim)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(enc_emb_dim, enc_hid_dim, bidirectional=True)
        self.lin = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src, src_len):
        src = self.emb(src)
        src = self.drop(src)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)
        packed_outs, hids = self.gru(packed_embedded)

        outs, _ = nn.utils.rnn.pad_packed_sequence(packed_outs)

        hid = torch.cat((hids[-2, :, :], hids[-1, :, :]), dim=1)
        hid = self.lin(hid)
        hid = torch.tanh(hid)

        return outs, hid
