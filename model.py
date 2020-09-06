import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from reformer_pytorch import Reformer
from transformers import *
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.linear1 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.linear2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.layernorm2 = nn.LayerNorm(config['hidden_size'])
    def forward(self, s1, s2):
        s1_embedding = self.linear1(s1)
        s2_embedding = self.linear2(s2)
        
        s2_embedding = F.relu(s2_embedding + s2)
        s2_embedding = self.layernorm2(s2_embedding)
        
        s12_similarity = F.relu(torch.matmul(s1, s2.transpose(1,2)))
        s12_attention = F.softmax(s12_similarity, dim=0).transpose(1, 2)
        s1_att_embedding = torch.matmul(s12_attention, s1_embedding)
        
        cross_embedding = torch.cat([s1_att_embedding, s2_embedding], dim=2)
        return cross_embedding
    
class CrossReformer(nn.Module):
    def __init__(self, config):
        super(CrossReformer, self).__init__()
        self.embedding = nn.Embedding(len(tokenizer.vocab), 768)
        self.rf = Reformer(
            dim = 768,
            depth = 1,
            max_seq_len = 4096,
            heads = 8,
            lsh_dropout = 0.1,
            causal = False
        )
        self.ca = CrossAttention(config)
        self.lstm = nn.LSTM(config['hidden_size'], config['hidden_size'],)
        self.word_attention = nn.Linear(768*2, 1)
        self.sentence_attention = nn.Linear(768*2, 1)
        self.classification_layer = nn.Linear(768*2, 3)
    def forward(self, article, wiki_datas, start_end_index, wiki_length, device):
        lf_article_word_embedding = self.embedding(article.unsqueeze(0))
        lf_article_word_embedding = self.rf(lf_article_word_embedding)
        all_sentence_embedding = []
        for index in start_end_index:
            start_index = index[0]
            end_index = index[1]
            sentence_embedding = lf_article_word_embedding[:,start_index:end_index,:]
            all_sentence_embedding.append(sentence_embedding)
        
        sentence_number = len(all_sentence_embedding)  
        
        all_wiki_data_embedding = []
        for wiki_data, length in zip(wiki_datas, wiki_length):
            lf_wiki_data_embedding = self.embedding(wiki_data.unsqueeze(0))
            lf_wiki_data_embedding, _ = self.lstm(lf_wiki_data_embedding.transpose(0, 1))
            all_wiki_data_embedding.append(lf_wiki_data_embedding.transpose(0, 1))
            
        all_final_embedding = []
        for sentence_embedding in all_sentence_embedding:
            sentence_wiki_embedding = []
            for wiki_data_embedding in all_wiki_data_embedding:
                ca_att_embedding = self.ca(wiki_data_embedding, sentence_embedding)
                word_level_attention = F.softmax(self.word_attention(ca_att_embedding), dim=1)
                ca_att_embedding = (ca_att_embedding * word_level_attention).sum(dim=1)
                sentence_wiki_embedding.append(ca_att_embedding)
            sentence_wiki_embedding = torch.cat(sentence_wiki_embedding, dim=0)
            sentence_level_attention = F.softmax(self.sentence_attention(sentence_wiki_embedding), dim=0)
            sentence_wiki_embedding = (sentence_wiki_embedding * sentence_level_attention).sum(dim=0).unsqueeze(0)
            all_final_embedding.append(sentence_wiki_embedding)
        all_final_embedding = torch.cat(all_final_embedding, dim=0)
        all_final_embedding = self.classification_layer(all_final_embedding)

        return all_final_embedding
    