import torch
from torch import nn
from torch.functional import F
from torch.nn.utils.rnn import pad_sequence
from .featpool import build_featpool  # downsample 1d temporal features to desired length
from .feat2d import \
    build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import build_contrastive_loss
from .loss import build_bce_loss
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv
import random


class NonLocalBlock(nn.Module):
    def __init__(self):
        super(NonLocalBlock, self).__init__()
        self.idim = 512
        self.odim = 512
        self.nheads = 8

        self.use_bias = True
        self.c_lin = nn.Linear(self.idim, self.odim * 2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)

    def forward(self, m_feats):
        B, nseg, _ = m_feats.size()

        m_k = self.v_lin(self.drop(m_feats))  # [B,num_seg,*]
        m_trans = self.c_lin(self.drop(m_feats))  # [B,nseg,2*]
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)
        for i in range(self.nheads):
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i]
            m2m = mk_slice @ mq_slice.transpose(1, 2) / ((self.odim // self.nheads) ** 0.5)
            m2m_w = torch.nn.functional.softmax(m2m, dim=2)
            w_list.append(m2m_w)
            r = m2m_w @ mv_slice if (i == 0) else torch.cat((r, m2m_w @ mv_slice), dim=2)

        updated_m = m_feats + r
        return updated_m


class LocalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super(LocalSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        num_windows = seq_len // self.window_size
        residual = seq_len % self.window_size
        if residual > 0:
            num_windows += 1
        outputs = []
        for i in range(num_windows):
            window_start = i * self.window_size
            window_end = min(window_start + self.window_size, seq_len)
            window_query = query[:, window_start:window_end, :]
            window_key = key[:, window_start:window_end, :]
            window_value = value[:, window_start:window_end, :]
            attn_weights = torch.matmul(window_query, window_key.transpose(-1, -2))
            attn_weights = attn_weights * self.scale
            attn_mask = torch.tril(torch.ones((window_end - window_start, window_end - window_start)))
            attn_mask = attn_mask.to(x.device)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            window_output = torch.matmul(attn_weights, window_value)
            outputs.append(window_output)
        outputs = torch.cat(outputs, dim=1)
        outputs = self.out(outputs)
        return outputs + x


class SecondFuse(nn.Module):
    def __init__(self):
        super(SecondFuse, self).__init__()
        self.vis_linear_b2 = nn.Linear(512, 512)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(True)
        self.avg_pool = nn.AvgPool1d(kernel_size=16, stride=16)

    def forward(self, boundary_map, local_map, content_map, map_mask, vis_h):
        vis_pool = self.avg_pool(vis_h)
        vis_pool = vis_pool.squeeze(-1)

        vis_h_b2 = self.vis_linear_b2(vis_pool)[:, :, None, None]  # f_s: B, C, 1, 1
        gate_b2 = torch.sigmoid(boundary_map * vis_h_b2)  # gate_b3: B, C, T, T
        gate_b3 = torch.sigmoid(gate_b2 * local_map)
        fused_h = gate_b3 * content_map * map_mask

        return fused_h


class ThirdFuse(nn.Module):
    def __init__(self):
        super(ThirdFuse, self).__init__()
        self.txt_linear_b1 = nn.Linear(512, 512)
        self.vis_linear_b2 = nn.Linear(512, 512)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(True)
        self.avg_pool = nn.AvgPool1d(kernel_size=16, stride=16)
        self.pred_layer = nn.Conv2d(512, 1, 1, 1)

    def forward(self, boundary_map, local_map, content_map, txt_h):
        txt_h_b1 = self.txt_linear_b1(txt_h)[:, :, None, None]  # f_s: N, C, 1, 1
        gate_b1 = torch.sigmoid(boundary_map * txt_h_b1)  # gate_b1: B, C, T, T
        gate_b2 = torch.sigmoid(gate_b1 * local_map)  # query-related content representation: B, C, T, T
        fused_h = gate_b2 * content_map

        iou_score = self.pred_layer(fused_h).squeeze(1)
        return iou_score


class SMTAN(nn.Module):
    def __init__(self, cfg):
        super(SMTAN, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg)
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.SMTAN.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.SMTAN.TEXT_ENCODER.NAME
        self.use_score_map_loss = cfg.MODEL.SMTAN.LOSS.USE_SCORE_MAP_LOSS
        self.cfg = cfg.MODEL.SMTAN
        self.thresh = cfg.MODEL.SMTAN.LOSS.THRESH
        self.w = self.cfg.RESIDUAL
        self.fuse_layer = SecondFuse()
        self.third_fuse = ThirdFuse()
        self.globle_attention = NonLocalBlock()
        self.local_attention = LocalSelfAttention(512, 8, 4)

    def forward(self, batches, cur_epoch=1):
        ious2d = batches.all_iou2d
        assert len(ious2d) == batches.feats.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            assert iou.size(0) == batches.num_sentence[idx]

        feats = self.featpool(batches.feats)  # feats [B C T]  [1 512 64]
        feats = feats.transpose(1, 2)
        feats_1 = self.globle_attention(feats)
        feats_2 = self.local_attention(feats)
        feats = (feats_1 + feats_2) / 2
        feats = feats.transpose(1, 2)

        boundary_map2d, local_map2d, content_map2d, mask2d = self.feat2d(feats)  # [B, 512, 64, 64]
        map2d = self.fuse_layer(boundary_map2d, local_map2d, content_map2d, mask2d, feats)
        map2d, map2d_iou = self.proposal_conv(map2d)
        sent_feat, sent_feat_iou = self.text_encoder(batches.queries, batches.wordlens)
        contrastive_scores = []
        iou_scores = []

        _, T, _ = map2d[0].size()
        for i, sf_iou in enumerate(sent_feat_iou):
            vid_feat_iou = map2d_iou[i]  # C x T x T
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou_norm = F.normalize(sf_iou, dim=1)  # num_sent x C
            iou_score1 = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T,
                                                                                                                 T)  # num_sent x T x T

            boundary_map = boundary_map2d[i]
            content_map = content_map2d[i]
            boundary_map = boundary_map.unsqueeze(0)
            local_map = boundary_map
            content_map = content_map.unsqueeze(0)
            iou_score2 = self.third_fuse(boundary_map, local_map, content_map, sf_iou_norm)
            iou_scores.append(((iou_score1 + iou_score2) * 10).sigmoid() * self.feat2d.mask2d)

        scoremap_loss_pos = torch.tensor(0.0).cuda()
        scoremap_loss_neg = torch.tensor(0.0).cuda()
        scoremap_loss_exc = torch.tensor(0.0).cuda()

        loss_iou_stnc = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
        loss_iou_phrase = torch.tensor(0.0).cuda()
        loss_vid, loss_sent = self.contrastive_loss(map2d, sent_feat, ious2d, None, None, batches.moments)

        if self.training:
            return loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc
        else:
            loss = (loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg,
                    scoremap_loss_exc)
            for i, sf in enumerate(sent_feat):
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)

                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T,
                                                                                                           T) * self.feat2d.mask2d  # num_sent x T x T

                boundary_map = boundary_map2d[i]
                content_map = content_map2d[i]
                boundary_map = boundary_map.unsqueeze(0)
                local_map = boundary_map
                content_map = content_map.unsqueeze(0)
                contrastive_score2 = self.third_fuse(boundary_map, local_map, content_map, sf_norm)

                contrastive_score = contrastive_score + contrastive_score2
                contrastive_scores.append(contrastive_score)

            return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores, loss

