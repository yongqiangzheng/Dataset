# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/3 15:02
# @Author  : ZYQ
# @File    : main.py
# @Software: PyCharm

import benepar
import copy
import os
import re
import spacy
import xml.etree.ElementTree as ET

from collections import defaultdict
from tqdm import tqdm
from spacy.symbols import ORTH


class Dataset4ABSA:
    def __init__(self, domain, dataset):
        self.dir_path = './xml_v2'
        self.domain = domain
        self.dataset = dataset
        self.all_dataset_path = {'lap14': {'train': self.dir_path + '/' + 'Semeval2014/Laptops_Train.xml',
                                           'test': self.dir_path + '/' + 'Semeval2014/Laptops_Test.xml'},
                                 'rest14': {'train': self.dir_path + '/' + 'SemEval2014/Restaurants_Train.xml',
                                            'test': self.dir_path + '/' + 'SemEval2014/Restaurants_Test.xml'},
                                 'rest15': {'train': self.dir_path + '/' + 'SemEval2015/Restaurants_Train.xml',
                                            'test': self.dir_path + '/' + 'SemEval2015/Restaurants_Test.xml'},
                                 'rest16': {'train': self.dir_path + '/' + 'SemEval2016/Restaurants_Train.xml',
                                            'test': self.dir_path + '/' + 'SemEval2016/Restaurants_Test.xml'},
                                 }
        self.sentiment_dict = {'negative': '-1', 'neutral': '0', 'positive': '1'}

    def parse_semeval14(self):
        sentence_list, aspect_list, label_list, aspect_index_list, sentence_id_list = [], [], [], [], []
        tree = ET.parse(self.all_dataset_path[self.domain][self.dataset])  # 解析树
        root = tree.getroot()  # 根节点
        sent_id = 0
        for sentence in tqdm(root.findall('sentence')):
            aspectTerms = sentence.find('aspectTerms')
            if aspectTerms is None:  # 去掉没有aspect的句子
                continue
            text = sentence.find('text').text  # 句子
            for aspectTerm in aspectTerms.findall('aspectTerm'):  # 遍历所有的aspect
                polarity = aspectTerm.get('polarity').strip()
                if polarity == 'conflict':  # 去掉conflict情感的句子
                    continue
                aspect = aspectTerm.get('term')
                start = aspectTerm.get('from')
                end = aspectTerm.get('to')
                assert text[int(start):int(end)] == aspect
                sentence_list.append(text)
                aspect_list.append(aspect)
                label_list.append(self.sentiment_dict[polarity])
                aspect_index_list.append([int(start), int(end)])
                sentence_id_list.append(sent_id)
            sent_id += 1
        return sentence_list, aspect_list, label_list, aspect_index_list, sentence_id_list

    def parse_semeval1516(self):
        sentence_list, aspect_list, label_list, aspect_index_list, sentence_id_list = [], [], [], [], []
        tree = ET.parse(self.all_dataset_path[self.domain][self.dataset])  # 解析树
        root = tree.getroot()  # 根节点
        sent_id = 0
        for review in tqdm(root.findall('Review')):
            for sentences in review.findall('sentences'):
                for sentence in sentences.findall('sentence'):
                    text = sentence.find('text').text  # 句子
                    if not sentence.findall('Opinions'):  # 删除没有aspect的句子
                        continue
                    for opinions in sentence.findall('Opinions'):
                        for opinion in opinions.findall('Opinion'):  # 遍历所有的aspect
                            aspect = opinion.get('target')
                            if aspect == 'NULL':
                                continue
                            polarity = opinion.get('polarity').strip()
                            start = opinion.get('from')
                            end = opinion.get('to')
                            assert text[int(start):int(end)] == aspect
                            sentence_list.append(text)
                            aspect_list.append(aspect)
                            label_list.append(self.sentiment_dict[polarity])
                            aspect_index_list.append([int(start), int(end)])
                            sentence_id_list.append(sent_id)
                    sent_id += 1
        return sentence_list, aspect_list, label_list, aspect_index_list, sentence_id_list

    def regex(self, string1):
        string1 = re.sub(r' ', '', string1)
        string1 = re.sub(r'-', ' - ', string1)
        string1 = re.sub(r'/', ' / ', string1)
        string1 = re.sub(r';', ' ; ', string1)
        string2 = re.sub(r' {2,}', ' ', string1)
        return string2

    def write_dataset(self, data, mode='one'):
        if mode == 'one':
            fout = open('one_sentence_one_aspect/{}_{}'.format(self.domain, self.dataset), 'w')
            nlp = spacy.load('en_core_web_trf')
            special_case = [{ORTH: "$T$"}]
            nlp.tokenizer.add_special_case("$T$", special_case)
            for sentence, aspect, label, aspect_index in tqdm(zip(data[0], data[1], data[2], data[3]),
                                                              total=len(data[0])):
                mask_sentence = sentence[:aspect_index[0]] + ' $T$ ' + sentence[aspect_index[1]:]
                sentence_tokens = [w.text for w in nlp(mask_sentence)]
                new_sentence = (' '.join(sentence_tokens)).strip()
                aspect_tokens = [a.text for a in nlp(aspect)]
                new_aspect = (' '.join(aspect_tokens)).strip()
                # regex
                new_sentence = self.regex(new_sentence)
                new_aspect = self.regex(new_aspect)
                fout.write(new_sentence + '\n' + new_aspect + '\n' + label + '\n')
            fout.close()
            return
        elif mode == 'all':
            if not os.path.exists('one_sentence_one_aspect/{}_{}'.format(self.domain, self.dataset)):
                print("run write_dataset(data, mode='one') first")
                return
            fin = open('one_sentence_one_aspect/{}_{}'.format(self.domain, self.dataset), 'r')
            fout = open('one_sentence_all_aspect/{}_{}'.format(self.domain, self.dataset), 'w')
            lines = fin.readlines()
            fin.close()

            sentence_list, aspect_list, label_list, aspect_index_list = [], [], [], []
            for i in range(0, len(lines), 3):
                text_left, _, text_right = lines[i].strip().partition('$T$')
                aspect = lines[i + 1].strip()
                label = lines[i + 2].strip()
                text = text_left + aspect + text_right
                sentence_list.append(text)
                aspect_list.append(aspect)
                label_list.append(label)
                aspect_index_list.append([len(text_left.split()), len(text_left.split()) + len(aspect.split())])
            compare_sent = sentence_list[0]
            compare_id = 0
            id2sent = {}
            # 检查同一句子下不同aspect的句子是否一致
            for sent, sent_id in tqdm(zip(sentence_list, data[4]), total=len(sentence_list)):
                id2sent[sent_id] = sent
                if sent_id == compare_id:
                    if sent == compare_sent:
                        compare_sent = sent
                        compare_id = sent_id
                    else:
                        print('-' * 10 + 'You may need to revise these sentences manually')
                        print(compare_sent)
                        print(sent)
                else:
                    compare_sent = sent
                    compare_id = sent_id
            id2aspect = defaultdict(lambda: [])
            for aspect, label, aspect_index, sent_id in tqdm(zip(aspect_list, label_list, aspect_index_list, data[4]),
                                                             total=len(aspect_list)):
                id2aspect[sent_id].append([aspect, label, aspect_index])
            aspect_and_index_list = []
            for id, aspects in id2aspect.items():
                for i in range(len(aspects)):
                    other_aspect = copy.deepcopy(aspects)
                    other_aspect.pop(i)
                    cur_aspect = aspects[i]
                    final_aspect = [cur_aspect] + other_aspect
                    aspect_and_index_list.append(final_aspect)

            assert len(sentence_list) == len(aspect_and_index_list)
            for sent, ai in zip(sentence_list, aspect_and_index_list):
                all_aspect_info = ''
                all_aspect_label = ''
                for token in ai:
                    aspect = token[0]
                    label = token[1]
                    start_index = token[2][0]
                    end_index = token[2][1]
                    all_aspect_info += aspect + '#' + str(start_index) + '#' + str(end_index) + ' /// '
                    all_aspect_label += label + ' '
                fout.write(sent + '\n' + all_aspect_info.strip() + '\n' + all_aspect_label.strip() + '\n')
            fout.close()
        else:
            print("try 'one' or 'all'")
            return

    def consitituency_parsing(self):
        pass


lap14_train = Dataset4ABSA('lap14', 'train')
data = lap14_train.parse_semeval14()
# lap14_train.write_dataset(data, 'one')
lap14_train.write_dataset(data, 'all')


lap14_test = Dataset4ABSA('lap14', 'test')
data = lap14_test.parse_semeval14()
# lap14_test.write_dataset(data, 'one')
lap14_test.write_dataset(data, 'all')

rest14_train = Dataset4ABSA('rest14', 'train')
data = rest14_train.parse_semeval14()
# rest14_train.write_dataset(data, 'one')
rest14_train.write_dataset(data, 'all')

rest14_test = Dataset4ABSA('rest14', 'test')
data = rest14_test.parse_semeval14()
# rest14_test.write_dataset(data, 'one')
rest14_test.write_dataset(data, 'all')

rest15_train = Dataset4ABSA('rest15', 'train')
data = rest15_train.parse_semeval1516()
# rest15_train.write_dataset(data, 'one')
rest15_train.write_dataset(data, 'all')

rest15_test = Dataset4ABSA('rest15', 'test')
data = rest15_test.parse_semeval1516()
# rest15_test.write_dataset(data, 'one')
rest15_test.write_dataset(data, 'all')

rest16_train = Dataset4ABSA('rest16', 'train')
data = rest16_train.parse_semeval1516()
# rest16_train.write_dataset(data, 'one')
rest16_train.write_dataset(data, 'all')

rest16_test = Dataset4ABSA('rest16', 'test')
data = rest16_test.parse_semeval1516()
# rest16_test.write_dataset(data, 'one')
rest16_test.write_dataset(data, 'all')
