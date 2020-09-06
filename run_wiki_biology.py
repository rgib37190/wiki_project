import re
import os
import pickle
import random
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from util import *
from transformers import *
from model import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
pd.options.display.max_rows = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

en_text = pd.read_csv('text_en.csv')
en_claim = pd.read_csv('claim_en.csv')
en_total_data = en_claim.merge(en_text, on='Qid', how='left')
en_total_data = en_total_data.dropna(subset=['value_label'])
en_total_data['wikidata_as_text'] = en_total_data['page_title_x'] + ' ' + en_total_data['prop_label'] + ' ' + en_total_data['value_label']

en_total_data['plain_text'] = en_total_data['plain_text'].apply(lambda x : x.lower())
en_total_data['wikidata_as_text'] = en_total_data['wikidata_as_text'].apply(lambda x : x.lower())
en_total_data['value_label'] = en_total_data['value_label'].apply(lambda x : x.lower().replace("\\",''))

# select 100 article
select_page_id = en_total_data['page_id'].unique()[:100]
en_total_data = en_total_data.loc[en_total_data['page_id'].isin(select_page_id)]

# load english model
seq_len = 4096
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load('en_core_web_sm')
page_id_record_list = []
all_article_list = []
all_start_end_index = []
all_start_pos_index = []
all_end_pos_index = []
all_article_change_value = []
all_article_wikidata_text = []
all_article_wikidata_text_length = []
all_label_list = []
for index, row in tqdm(en_total_data.iterrows(), total=len(en_total_data), desc='Preprocessing...'):
    page_id = row['page_id']
    if page_id not in page_id_record_list:
        text = row['plain_text']
        spacy_text = nlp(text)
        prop_values = en_total_data.loc[en_total_data['page_id']==page_id, 'value_label']
        props = en_total_data.loc[en_total_data['page_id']==page_id, 'prop']
        wikidata_texts = en_total_data.loc[en_total_data['page_id']==page_id, 'wikidata_as_text'].unique()
        
        start_index = 0
        article = []
        start_end_index = []
        start_pos_index = []
        end_pos_index = []
        article_change_value = []
        select_wikidata_text = []
        prop_label_list = []
        prop_value_list = []
        label_list = []
        segment_sentence_number = 0
        segment_sentence = ''
        for sentence in spacy_text.sents:
            sentence = str(sentence).strip()
            if segment_sentence_number < 4:
                segment_sentence = segment_sentence + sentence
                segment_sentence_number += 1
            elif segment_sentence_number == 4:
                candidate_changed_sentences = []
                candidate_no_changed_sentence = []
                if 'category' not in segment_sentence:
                    for prop, prop_value, wikidata_text in zip(props,prop_values,wikidata_texts):
                        pattern = re.compile(prop_value)
                        match_result = re.search(pattern,segment_sentence)
                        if match_result:
                            select_wikidata_text.append(wikidata_text)
                            disturb_choices = list(en_total_data.loc[en_total_data['prop']==prop, 'value_label'].unique())
                            disturb_choices.remove(prop_value)
                            if len(disturb_choices) == 0:
                                candidate_no_changed_sentence.append(segment_sentence)
                            else:
                                disturb_choice = random.sample(disturb_choices, 1)[0]
                                disturb_sentence = segment_sentence.replace(prop_value, disturb_choice)
                                candidate_changed_sentences.append(disturb_sentence)
                                article_change_value.append(disturb_choice)
                    if len(candidate_changed_sentences) != 0:
                        if random.random() < 0.5:
                            candidate_changed_sentence = random.sample(candidate_changed_sentences, 1)[0]
                            candidate_changed_sentence = clean_not_english_word(candidate_changed_sentence)
                            candidate_changed_sentence = clean_stopwords(candidate_changed_sentence)
                            tokenize_changed_sentence = tokenizer.encode(candidate_changed_sentence, add_special_tokens=False)
                            if len(tokenize_changed_sentence) != 0:
                                if (len(article) + len(tokenize_changed_sentence)) < seq_len:
                                    article.extend(tokenize_changed_sentence)
                                    start_end_index.append([start_index, start_index+len(tokenize_changed_sentence)])
                                    start_index = start_index + len(tokenize_changed_sentence)
                                    label_list.append(0)
                        else:
                            segment_sentence = clean_not_english_word(segment_sentence)
                            segment_sentence = clean_stopwords(segment_sentence)
                            tokenize_sentence = tokenizer.encode(segment_sentence, add_special_tokens=False)
                            if len(tokenize_sentence) != 0:
                                if (len(article) + len(tokenize_sentence)) < seq_len:
                                    article.extend(tokenize_sentence)
                                    start_end_index.append([start_index, start_index+len(tokenize_sentence)])
                                    start_index = start_index + len(tokenize_sentence)
                                    label_list.append(1)
                    elif len(candidate_no_changed_sentence) != 0:
                        candidate_no_changed_sentence = candidate_no_changed_sentence[0]
                        candidate_no_changed_sentence = clean_not_english_word(candidate_no_changed_sentence)
                        candidate_no_changed_sentence = clean_stopwords(candidate_no_changed_sentence)
                        tokenize_no_changed_sentence = tokenizer.encode(candidate_no_changed_sentence, add_special_tokens=False)
                        if len(tokenize_no_changed_sentence) != 0:
                            if (len(article) + len(tokenize_no_changed_sentence)) < seq_len:
                                article.extend(tokenize_no_changed_sentence)
                                start_end_index.append([start_index, start_index+len(tokenize_no_changed_sentence)])
                                start_index = start_index + len(tokenize_no_changed_sentence)
                                label_list.append(1)
                    else:
                        # irrelevant
                        segment_sentence = clean_not_english_word(segment_sentence)
                        segment_sentence = clean_stopwords(segment_sentence)
                        tokenize_sentence = tokenizer.encode(segment_sentence, add_special_tokens=False)
                        if len(tokenize_sentence) != 0:
                            if (len(article) + len(tokenize_sentence)) < seq_len:
                                article.extend(tokenize_sentence)
                                start_end_index.append([start_index, start_index+len(tokenize_sentence)])
                                start_index = start_index + len(tokenize_sentence)
                                label_list.append(2)
                    segment_sentence_number = 0
                    segment_sentence = ''
            else:
                pass

        page_id_record_list.append(page_id)
        
        select_wikidata_text = list(set(select_wikidata_text))
        tokenize_wiki_data = []
        tokenize_wiki_data_length = []
        for wiki_data in select_wikidata_text:
            wiki_data = tokenizer.encode(wiki_data, add_special_tokens=False)
            if len(wiki_data) != 0:
                tokenize_wiki_data.append(torch.tensor(wiki_data, device=device))
                tokenize_wiki_data_length.append(len(wiki_data))
        if len(tokenize_wiki_data) != 0:
            all_article_wikidata_text.append(tokenize_wiki_data)
            all_article_wikidata_text_length.append(tokenize_wiki_data_length)
            
            all_article_list.append(torch.tensor(article, device=device))
            all_start_end_index.append(start_end_index)
            all_article_change_value.append(list(set(article_change_value)))
            all_label_list.append(torch.tensor(label_list,device=device))
    else:
        pass

train_article, test_article, train_wiki_data, test_wiki_data, train_start_end_index, test_start_end_index, train_wiki_length, test_wiki_length, train_label, test_label = train_test_split(all_article_list, all_article_wikidata_text, all_start_end_index, all_article_wikidata_text_length, all_label_list, train_size=0.8,random_state=42)

train_article.append(torch.zeros(seq_len, device=device))
test_article.append(torch.zeros(seq_len, device=device))

train_article = pad_sequence(train_article, batch_first=True)[:-1]
test_article = pad_sequence(test_article, batch_first=True)[:-1]

config = {'hidden_size':768}

train_label_0 = 0
train_label_1 = 0
train_label_2 = 0
for i in train_label:
    for j in i:
        if j== 0:
            train_label_0 += 1
        elif j == 1:
            train_label_1 += 1
        else:
            train_label_2 += 1
class_weight = torch.tensor([1/train_label_0, 1/train_label_1, 1/train_label_2], device=device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cr = CrossReformer(config).cuda()
optimizer = optim.Adam(cr.parameters(), lr=5e-5, eps=1e-8)
CE = nn.CrossEntropyLoss(weight=class_weight)


epochs = 40
for epoch in tqdm(range(epochs), desc='Training...'):
    CE_mean_loss = []
    train_predict_result = []
    train_true_label = []
    cr.train()
    for article, wiki_datas, start_end_index, wiki_length, label in zip(train_article, train_wiki_data, train_start_end_index, train_wiki_length, train_label):
       
        output = cr(article, wiki_datas, start_end_index, wiki_length, device)
        softmax = nn.Softmax(dim=1)
        train_pred_prob = softmax(output)
        train_pred_result = np.argmax(train_pred_prob.cpu().detach().numpy(), axis=1)
        for result in train_pred_result:
            train_predict_result.append(result)
        for i in label.cpu().detach().numpy():
            train_true_label.append(i)
        CE_loss = CE(output, label)
        CE_mean_loss.append(CE_loss.cpu().detach().numpy())

        CE_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del output
    
    CE_mean_loss = np.mean(CE_mean_loss) 
    
    print('----------------------Train------------------------------')
    print(classification_report(train_true_label, train_predict_result))
    
    print('Epochs:{},CE Loss:{:5f}'.format(epoch,CE_mean_loss))
    
    if epoch == epochs - 1:
        model_save_path = '/home/champion/wiki/cr'
        save_model_name = 'cr_biology.pickle'.format(epoch)
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        model_structure_path = os.path.join(model_save_path, save_model_name)
        torch.save(cr.state_dict(), model_structure_path)


cr = CrossReformer(config).cuda()
pretrain_weight = torch.load('/home/champion/wiki/cr/cr_biology.pickle', map_location=device)
cr.load_state_dict(pretrain_weight)

test_predict_result = []
test_predict_prob = []
test_true_label = []
cr.eval()
for article, wiki_datas, start_end_index, length, label in zip(test_article, test_wiki_data, test_start_end_index, test_wiki_length, test_label):
    output = cr(article, wiki_datas, start_end_index, length, device)
    softmax = nn.Softmax(dim=1)
    test_pred_prob = softmax(output)
    test_pred_result = np.argmax(test_pred_prob.cpu().detach().numpy(), axis=1)
    test_pred_prob = torch.max(test_pred_prob, axis=1).values.cpu().detach().numpy()
    for result in test_pred_result:
        test_predict_result.append(result)
    for prob in test_pred_prob:
        test_predict_prob.append(prob)
    for i in label.cpu().detach().numpy():
        test_true_label.append(i)
    del output
print('----------------------Test------------------------------')
print(classification_report(test_true_label, test_predict_result))