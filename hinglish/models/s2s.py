import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_log_prob_obj(score, length, alpha=0.75):
    return score / length**alpha


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, pad_idx, sos_idx, eos_idx, device):
        super().__init__()

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

        self.enc = enc
        self.dec = dec

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def encoder_step(self, src, src_len):
        enc_outs, hid = self.enc(src, src_len)
        return enc_outs, hid

    def decoder_step(self, out, hid, enc_outs, mask):
        out, hid, attn = self.dec(out, hid, enc_outs, mask)
        out = F.log_softmax(out, dim=1)
        return out, hid


class Seq2SeqInfer(Seq2Seq):
    def prep_trg(self, src):
        max_len = 40
        trg = torch.zeros((max_len, src.shape[1])).long()
        trg.fill_(self.sos_idx).to(src.device)
        return trg

    def get_top_k(self, curr_out, beam):
        if beam == 1:
            topk_items = curr_out.max(1)
            topk_items = zip(topk_items[1], topk_items[0])
        else:
            topk_items = curr_out.topk(beam)
            topk_items = zip(topk_items[1].squeeze(), topk_items[0].squeeze())
        return topk_items

    def get_seq_score(self, prev_score, prob):
        return prev_score + prob

    def beam_decode_step(self, seq, enc_outs, mask, beam):
        new_seqs = []
        out = seq[0]
        hid = seq[1]
        prev_outs = seq[2]
        score = seq[3]
        # print(out)
        if out != self.eos_idx:
            curr_out, curr_hid = self.decoder_step(out, hid, enc_outs, mask)
            topk_items = self.get_top_k(curr_out, beam)

            for pred, prob in topk_items:
                pred = torch.tensor([pred.clone()])
                # print(_, f"in:{out}, out:{pred}, prob:{prob}")
                all_outs = prev_outs + [pred]
                new_score = self.get_seq_score(score, prob)
                new_seqs.append([pred, curr_hid, all_outs, new_score])
        else:
            # print(_, f"in:{out}, out:{out}, prob:{1}")
            new_seqs.append(seq)

        return new_seqs

    def select_beam(self, sequences, beam):
        sequences = sorted(
            sequences,
            key=lambda seq: norm_log_prob_obj(seq[-1], len(seq[2])),
            reverse=True
        )
        sequences = sequences[:beam]
        return sequences

    def beam_decode_loop(self, trg, out, hid, enc_outs, mask, beam):
        """
        Just for inference
        """
        max_len = trg.shape[0]

        seqs = [[out, hid, [out], torch.tensor(0.0)]]
        for _ in range(0, max_len):
            new_seqs = []
            for seq in seqs:
                new_seqs += self.beam_decode_step(seq, enc_outs, mask, beam)

            new_seqs = self.select_beam(new_seqs, beam)
            del seqs
            seqs = new_seqs
            del new_seqs

        outs = []
        for i in range(beam):
            seqs[i][2] = torch.cat(seqs[i][2], 0)
            score = norm_log_prob_obj(seqs[i][-1].item(), len(seqs[i][2]))
            outs.append((seqs[i][2], score))

        return outs

    def forward(self, src, src_len, beam=1):
        trg = self.prep_trg(src)

        enc_outs, hid = self.encoder_step(src, src_len)
        mask = self.create_mask(src)
        out = trg[0, :]

        outs = self.beam_decode_loop(trg, out, hid, enc_outs, mask, beam)
        return outs


class Seq2SeqTrain(Seq2Seq):

    def simple_decode_loop(self, trg, out, hid, enc_outs, mask, tf_ratio, batch_size):
        """
        Just for inference
        """
        max_len = trg.shape[0]
        trg_vocab_dim = self.dec.vocab_size

        outs = torch.zeros(max_len, batch_size, trg_vocab_dim)
        outs = outs.to(self.device)
        outs[:, :, self.eos_idx] = 100

        for t in range(1, max_len):
            out, hid = self.decoder_step(out, hid, enc_outs, mask)
            outs[t - 1] = out

            tf_flag = random.random() < tf_ratio
            top1 = out.max(1)[1]
            out = (trg[t] if tf_flag else top1)

        return outs

    def forward(self, src, src_len, trg, tf_ratio=0.5):
        batch_size = src.shape[1]

        enc_outs, hid = self.encoder_step(src, src_len)
        mask = self.create_mask(src)
        out = trg[0, :]

        outs = self.simple_decode_loop(
            trg, out, hid, enc_outs, mask, tf_ratio, batch_size)
        return outs
