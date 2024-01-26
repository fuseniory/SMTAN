import torch
from torch.functional import F
from smtan.data.datasets.utils import box_iou


class BceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.hinge_loss = False

    def linear_scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d, epoch):
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        loss = 0
        iou1d = self.linear_scale(iou1d).clamp(0, 1)
        loss += self.bceloss(scores1d, iou1d).mean()
        return loss


def build_bce_loss(cfg, mask2d):
    min_iou = cfg.MODEL.SMTAN.LOSS.MIN_IOU
    max_iou = cfg.MODEL.SMTAN.LOSS.MAX_IOU
    return BceLoss(min_iou, max_iou, mask2d)


class PhraseLoss(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.sent_neg_iou = cfg.MODEL.SMTAN.LOSS.SENT_NEG_IOU

    def __call__(self, feat2ds, phr_feats, iou2ds, gt_proposals):
        B, C, _, _ = feat2ds.size()
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1)
        feat1ds_norm = F.normalize(feat1ds, dim=1)

        for i, (phr_feat, iou2d, gt_per_video) in enumerate(zip(phr_feats, iou2ds, gt_proposals)):
            feat1d = feat1ds_norm[i, :, :]
            phr_feat = torch.stack(phr_feat)
            sent_feat = phr_feat[:, 0, :]
            phr_feat = phr_feat[:, 1:, :]
            iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1)

            num_sent, num_phr, _ = phr_feat.size()
            phr_feat = phr_feat.reshape(num_sent * num_phr, -1)
            phr_feat = F.normalize(phr_feat, dim=1)
            phr_score = torch.mm(phr_feat, feat1d).reshape(num_sent, num_phr, -1)
            phr_feat = phr_feat.reshape(num_sent, num_phr, -1)

            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_neg_iou
            sent_neg_mask = iou_mask.float()

            for sen1 in range(num_sent):
                for sen2 in range(sen1 + 1, num_sent):
                    if iou_map_per_video[sen1, sen2] > self.sent_neg_iou:
                        pos_loss = self.pos_phr_loss(phr_score[sen1, :, :], phr_score[sen2, :, :], phr_feat[sen1, :, :],
                                                     phr_feat[sen2, :, :])


class ContrastiveLoss(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.T_v = cfg.MODEL.SMTAN.LOSS.TAU_VIDEO
        self.T_s = cfg.MODEL.SMTAN.LOSS.TAU_SENT
        self.cri = torch.nn.CrossEntropyLoss()
        self.neg_iou = cfg.MODEL.SMTAN.LOSS.NEGATIVE_VIDEO_IOU
        self.top_k = cfg.MODEL.SMTAN.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL
        self.sent_removal_iou = cfg.MODEL.SMTAN.LOSS.SENT_REMOVAL_IOU
        self.margin = cfg.MODEL.SMTAN.LOSS.MARGIN
        self.eps = 1e-6
        self.dataset = cfg.DATASETS.NAME

    def __call__(self, feat2ds, sent_feats, iou2ds, phr_feats, phr_weights, gt_proposals):
        B, C, _, _ = feat2ds.size()
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1)
        feat1ds_norm = F.normalize(feat1ds, dim=1)
        sent_feat_cat = torch.cat(sent_feats, 0)
        sum_num_sent = sent_feat_cat.size(0)
        sent_feat_cat_norm = F.normalize(sent_feat_cat, dim=1)
        sent_mask = torch.ones(sum_num_sent, sum_num_sent, device=feat2ds.device)

        all_num_sent = [0]
        curr_num_sent = 0
        for i in range(len(sent_feats)):
            curr_num_sent += sent_feats[i].size(0)
            all_num_sent.append(curr_num_sent)
        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou
            sent_mask[all_num_sent[i]:all_num_sent[i + 1], all_num_sent[i]:all_num_sent[i + 1]] = iou_mask.float()
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()
        margin_mask = torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)) * self.margin
        vid_pos_list = []
        vid_neg_list = []
        sent_pos_list = []
        sent_neg_list = []

        for i, (sent_feat, iou2d) in enumerate(zip(sent_feats, iou2ds)):
            num_sent_this_batch = sent_feat.size(0)
            feat1d = feat1ds_norm[i, :, :]
            sent_feat = F.normalize(sent_feat, dim=1)
            iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0),
                                                             -1)
            topk_index = torch.topk(iou1d, self.top_k, dim=-1)[1]
            selected_feat = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1,
                                                                                             self.top_k)
            selected_feat = selected_feat.permute(1, 2, 0)

            vid_pos = torch.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1,
                                                                self.top_k) - self.margin  
            vid_neg = torch.mm(selected_feat.view(-1, C),
                               sent_feat_cat_norm.t()).reshape(-1, self.top_k,
                                                               sum_num_sent)  
            vid_pos_list.append(vid_pos)
            vid_neg_list.append(vid_neg)
            
            sent_pos_list.append(vid_pos.clone())
            sent_neg_same_video = torch.mm(sent_feat, feat1d)  
            iou_neg_mask = (
                        iou1d < self.neg_iou).float()  
            sent_neg_same_video = iou_neg_mask * sent_neg_same_video  
            feat1d_other_video = feat1ds_norm.index_select(dim=0, index=torch.arange(
                B, device=feat2ds.device)[
                torch.arange(B, device=feat2ds.device) != i])  
            feat1d_other_video = feat1d_other_video.transpose(1, 0).reshape(C,
                                                                            -1)  
            sent_neg_other_video = torch.mm(sent_feat,
                                            feat1d_other_video)  
            sent_neg_all = [vid_pos.clone().unsqueeze(2),
                            sent_neg_same_video.unsqueeze(1).repeat(1, self.top_k, 1),
                            sent_neg_other_video.unsqueeze(1).repeat(1, self.top_k, 1)]
            sent_neg_list.append(torch.cat(sent_neg_all, dim=2))  
        
        
        vid_pos = (torch.cat(vid_pos_list, dim=0).transpose(0, 1)) / self.T_v  
        vid_neg = torch.cat(vid_neg_list, dim=0).permute(1, 0,
                                                         2)  
        vid_neg = (
                              vid_neg - margin_mask) / self.T_v  
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()
        vid_neg_exp = torch.exp(vid_neg) * sent_mask.clamp(min=0, max=1)
        loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=2, keepdim=False))).mean()
        sent_pos = torch.cat(sent_pos_list, dim=0) / self.T_s
        sent_neg = torch.cat(sent_neg_list, dim=0) / self.T_s
        sent_neg_exp = torch.exp(sent_neg)
        loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean()
        return loss_vid, loss_sent


def build_contrastive_loss(cfg, mask2d):
    return ContrastiveLoss(cfg, mask2d)
