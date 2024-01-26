import torch
from torch import nn
from transformers import DistilBertModel
from torch.functional import F
from smtan.data.datasets.utils import bert_embedding_batch
from transformers import DistilBertTokenizer


class AttentivePooling(nn.Module):
    def __init__(self, feat_dim, att_hid_dim):
        super(AttentivePooling, self).__init__()
        self.feat_dim = feat_dim
        self.att_hid_dim = att_hid_dim
        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        self.to_alpha = nn.Linear(self.att_hid_dim, 1, bias=False)

    def forward(self, feats, global_feat, f_masks=None):
        assert len(feats.size()) == 3
        assert len(global_feat.size()) == 2
        assert f_masks is None or len(f_masks.size()) == 2
        attn_f = self.feat2att(feats)
        dot = torch.tanh(attn_f)        
        alpha = self.to_alpha(dot)      
        if f_masks is not None:
            alpha = alpha.masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
        attw =  F.softmax(alpha, dim=1) 
        attw = attw.squeeze(-1)
        return attw


class DistilBert(nn.Module):
    def __init__(self, joint_space_size, dataset, use_phrase, drop_phrase):
        super().__init__()
        MODEL_PATH = "/bert_model/distilbert-base-uncased"
        self.bert = DistilBertModel.from_pretrained(MODEL_PATH)
        self.fc_out1 = nn.Linear(768, joint_space_size)
        self.fc_out2 = nn.Linear(768, joint_space_size)
        self.dataset = dataset
        self.layernorm = nn.LayerNorm(768)
        self.aggregation = "avg"  
        self.use_phrase = use_phrase
        self.drop_phrase = drop_phrase
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
        self.joint_space_size = joint_space_size
    
    def encode_single(self, query, word_len):
        N, word_length = query.size(0), query.size(1)
        attn_mask = torch.zeros(N, word_length, device=query.device)
        for i in range(N):
            attn_mask[i, :word_len[i]] = 1  
        bert_encoding = self.bert(query, attention_mask=attn_mask)[0]
        if self.aggregation == "cls":
            pass
        elif self.aggregation == "avg":
            avg_mask = torch.zeros(N, word_length, device=query.device)
            for i in range(N):
                avg_mask[i, :word_len[i]] = 1       
            avg_mask = avg_mask / (word_len.unsqueeze(-1))
            bert_encoding = bert_encoding.permute(2, 0, 1) * avg_mask  
            query = bert_encoding.sum(-1).t()  
            query = self.layernorm(query)
            out_iou = self.fc_out1(query)
            out = self.fc_out2(query)
        else:
            raise NotImplementedError
        return out, out_iou

    def forward(self, queries, wordlens):
        sent_feat = []
        sent_feat_iou = []
        for query, word_len in zip(queries, wordlens):  
            out, out_iou = self.encode_single(query, word_len)
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        return sent_feat, sent_feat_iou
    def encode_sentences(self, sentences):
        sent_feat = []
        sent_feat_iou = []
        stnc_query, stnc_len = bert_embedding_batch(sentences, self.tokenizer)
        for query, word_len in zip(stnc_query, stnc_len):  
            out, out_iou = self.encode_single(query.cuda(), word_len.cuda())
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        return sent_feat, sent_feat_iou


def build_text_encoder(cfg):
    joint_space_size = cfg.MODEL.SMTAN.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME 
    use_phrase = cfg.MODEL.SMTAN.TEXT_ENCODER.USE_PHRASE
    drop_phrase = cfg.MODEL.SMTAN.TEXT_ENCODER.DROP_PHRASE
    return DistilBert(joint_space_size, dataset_name, use_phrase, drop_phrase)

