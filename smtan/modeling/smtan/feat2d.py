import torch
from torch import nn


class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c):
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
            )
        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers
        self.idim = 512
        self.odim = 512
        self.nheads = 8
        self.use_bias = True
        self.c_lin = nn.Linear(self.idim, self.odim * 2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        m_feats = x.transpose(1, 2)
        B, D, N = x.shape
        boundary_map2d = x.new_zeros(B, D, N, N)
        local_map2d = x.new_zeros(B, D, N, N)
        content_map2d = x.new_zeros(B, D, N, N)
        mask2d = self.mask2d.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        content_map2d[:, :, range(N), range(N)] = x
        local_map2d[:, :, range(N), range(N)] = x
        boundary_map2d[:, :, range(N), range(N)] = x
        for (i, j) in self.maskij:
            i_mask = x[:, :, i]
            i_mask = i_mask.transpose(1, 2)
            i_mask = i_mask.transpose(1, 2)
            j_mask = x[:, :, j]
            j_mask = j_mask.transpose(1, 2)
            j_mask = j_mask.transpose(1, 2)
            boundary_map2d[:, :, i, j] = (i_mask + j_mask) / 2
        for (i, j) in self.maskij:
            m = list(i)
            n = list(j)
            k = [(int)((a + b) / 2) for a, b in zip(m, n)]
            i_mask = x[:, :, i]
            k_mask = x[:, :, k]
            j_mask = x[:, :, j]
            local_map2d[:, :, i, j] = (i_mask + j_mask + 0.5 * k_mask) / 2.5
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            content_map2d[:, :, i, j] = x
        B, nseg, _ = m_feats.size()
        m_k = self.v_lin(self.drop(m_feats))
        m_trans = self.c_lin(self.drop(m_feats))
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)
        m2m = m_k @ m_q.transpose(1, 2) / ((self.odim // self.nheads) ** 0.5)
        m2m_w = torch.nn.functional.softmax(m2m, dim=2)
        m2m_w = m2m_w.unsqueeze(1)
        boundary_map2d = boundary_map2d + boundary_map2d * m2m_w
        local_map2d = local_map2d + local_map2d * m2m_w
        content_map2d = content_map2d + content_map2d * m2m_w
        return boundary_map2d, local_map2d, content_map2d, mask2d

def build_feat2d(cfg):
    pooling_counts = cfg.MODEL.SMTAN.FEAT2D.POOLING_COUNTS
    num_clips = cfg.MODEL.SMTAN.NUM_CLIPS
    hidden_size = cfg.MODEL.SMTAN.FEATPOOL.HIDDEN_SIZE
    if cfg.MODEL.SMTAN.FEAT2D.NAME == "conv":
        return SparseConv(pooling_counts, num_clips, hidden_size)
    elif cfg.MODEL.SMTAN.FEAT2D.NAME == "pool":
        return SparseMaxPool(pooling_counts, num_clips)
    else:
        raise NotImplementedError("No such feature 2d method as %s" % cfg.MODEL.SMTAN.FEAT2D.NAME)
