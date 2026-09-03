import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from attention import FeedForward
from ban_2 import BANLayer
from models_5 import (
    MLPDecoder,
    ProteinCNN,
    RandomLayer,
    binary_cross_entropy,
    cross_entropy_logits,
    entropy_logits,
    mmd_loss,
)


ABLATION_NONE = "none"
ABLATION_NO_PEP_ENCODER = "no_pep_encoder"
ABLATION_NO_PAT_ENCODER = "no_pat_encoder"
ABLATION_NO_PAM = "no_pam"
SUPPORTED_ABLATIONS = {
    ABLATION_NONE,
    ABLATION_NO_PEP_ENCODER,
    ABLATION_NO_PAT_ENCODER,
    ABLATION_NO_PAM,
}


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.pre_softmax_talking_heads = nn.Conv2d(n_heads, n_heads, 1, bias=False)
        self.post_softmax_talking_heads = nn.Conv2d(n_heads, n_heads, 1, bias=False)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.head_dim = hid_dim // n_heads

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(bsz, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale
        energy = self.pre_softmax_talking_heads(energy)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        attention = self.post_softmax_talking_heads(attention)
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.hid_dim)
        return self.fc(x)


class PatEncoderAblation(nn.Module):
    """EncoderLayer variant used for PatEncoder/PAM ablation."""

    def __init__(self, hid_dim, kernel_size, dropout, use_pam=True):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.use_pam = use_pam
        self.sa = SelfAttention(hid_dim, 8, dropout)
        self.ea = SelfAttention(hid_dim, 8, dropout)
        self.ff = FeedForward(hid_dim, hid_dim, glu=True, dropout=dropout)
        self.do = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hid_dim)
        self.conv_net = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.Sigmoid(),
        )
        self.proj_t = nn.Linear(128, 128)
        self.proj_c = nn.Linear(128, 128)

    def forward(self, conv_input, usesrc=None):
        if self.use_pam:
            if usesrc is None:
                raise ValueError("usesrc is required when PAM is enabled")
            conv_input = self.ln(conv_input + self.do(self.ea(conv_input, usesrc, usesrc)))

        global_output = self.ln(conv_input + self.do(self.sa(conv_input, conv_input, conv_input)))
        local_output = conv_input.permute(0, 2, 1)
        local_output = self.conv_net(local_output).permute(0, 2, 1)
        gate_weight = self.gate(torch.cat([global_output, local_output], dim=-1))
        return gate_weight * self.proj_t(global_output) + (1 - gate_weight) * self.proj_c(local_output)


class DrugBAN(nn.Module):
    """
    DrugBAN with switchable ablations.

    Ablations:
    - none: original structure.
    - no_pep_encoder: remove ProteinCNN; protein feature is only the fc-projected ESM feature.
    - no_pat_encoder: remove EncoderLayer/PatEncoder; pathogen feature is only the fc-projected BioBERT feature.
    - no_pam: keep PepEncoder and PatEncoder, but remove the PAM path:
      no talking-src Conv1d/GLU and no external attention `ea(conv_input, usesrc, usesrc)`.
    """

    def __init__(self, ablation=ABLATION_NONE, **config):
        super().__init__()
        if ablation not in SUPPORTED_ABLATIONS:
            raise ValueError(f"Unsupported ablation '{ablation}'. Choose from {sorted(SUPPORTED_ABLATIONS)}")

        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        protein_padding = config["PROTEIN"]["PADDING"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]

        self.ablation = ablation
        self.use_pep_encoder = ablation != ABLATION_NO_PEP_ENCODER
        self.use_pat_encoder = ablation != ABLATION_NO_PAT_ENCODER
        self.use_pam = ablation != ABLATION_NO_PAM and self.use_pat_encoder

        self.protein_extractor = (
            ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
            if self.use_pep_encoder
            else None
        )
        self.drug_extractor = (
            PatEncoderAblation(hid_dim=128, kernel_size=3, dropout=0.2, use_pam=self.use_pam)
            if self.use_pat_encoder
            else None
        )

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=128, h_dim=mlp_in_dim, h_out=ban_heads),
            name="h_mat",
            dim=None,
        )
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        self.fc = nn.Linear(2, 128)
        self.ft = nn.Linear(32, 128)
        self.conv = nn.Conv1d(128, 128 * 2, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def _encode_peptide(self, v_p):
        bsz = v_p.shape[0]
        src = v_p.reshape(bsz, -1, 2)
        src = self.fc(src)
        if self.use_pep_encoder:
            _, src = self.protein_extractor(src)
        return src

    def _encode_pathogen(self, bg_d, src):
        bsz = bg_d.shape[0]
        trg = bg_d.reshape(bsz, -1, 32).to(src.device)
        trg = self.ft(trg)

        if not self.use_pat_encoder:
            return trg

        if self.use_pam:
            talkingsrc = torch.zeros_like(src)
            talkingsrc = torch.add(talkingsrc, src)
            t_talkingsrc = talkingsrc.permute(0, 2, 1)
            conved_t = self.conv(self.dropout(t_talkingsrc))
            conved_t = F.glu(conved_t, dim=1)
            usesrc = conved_t.permute(0, 2, 1)
            return self.drug_extractor(trg, usesrc)

        return self.drug_extractor(trg, usesrc=None)

    def forward(self, bg_d, v_p, mode="train"):
        src = self._encode_peptide(v_p)
        trg = self._encode_pathogen(bg_d, src)
        f, att = self.bcn(trg, src)
        score = self.mlp_classifier(f)
        return trg, src, f, score
