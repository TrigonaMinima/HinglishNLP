import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attention = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, enc_outs, dec_hid, mask):
        batch_size = enc_outs.shape[1]
        src_len = enc_outs.shape[0]

        dec_hid = dec_hid.unsqueeze(1).repeat(1, src_len, 1)
        enc_outs = enc_outs.permute(1, 0, 2)

        energy_input = torch.cat((dec_hid, enc_outs), dim=2)
        energy = self.attention(energy_input)
        energy = torch.tanh(energy)
        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)
