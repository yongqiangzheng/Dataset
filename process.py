# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/3 00:03
# @Author  : ZYQ
# @File    : process.py
# @Software: PyCharm

# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project:     ABSA2023
@Author:      ZYQ
@Date:        2022/10/28 15:34
@File name:   process.py
"""
import re
import xml.etree.ElementTree as ET
# import stanza
# import spacy
# from tqdm import tqdm


def xml_parse_14(infile, outfile):
    mask_sentence_list, aspect_list, label_list = [], [], []
    sentiment_dict = {'negative': '-1', 'neutral': '0', 'positive': '1'}
    tree = ET.parse(infile)  # 解析树
    root = tree.getroot()  # 根节点
    for sentence in root.findall('sentence'):
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:  # 去掉没有aspect的句子
            continue
        text = sentence.find('text').text  # 句子
        for aspectTerm in aspectTerms.findall('aspectTerm'):  # 遍历所有的aspect
            polarity = aspectTerm.get('polarity').strip()
            if polarity == 'conflict':  # 去掉conflict情感的句子
                continue
            start = aspectTerm.get('from')
            end = aspectTerm.get('to')
            mask_sentence_list.append((text[:int(start)] + ' $T$ ' + text[int(end):]).strip())
            aspect_list.append(aspectTerm.get('term').strip())
            label_list.append(sentiment_dict[polarity])
    fout = open(outfile, 'w', encoding='utf-8')
    for i, j, k in zip(mask_sentence_list, aspect_list, label_list):
        fout.write(i + '\n' + j + '\n' + k + '\n')
    fout.close()


def xml_parse_1516(infile, outfile):
    mask_sentence_list, aspect_list, label_list, start_list, end_list = [], [], [], [], []
    sentiment_dict = {'negative': '-1', 'neutral': '0', 'positive': '1'}
    tree = ET.parse(infile)  # 解析树
    root = tree.getroot()  # 根节点
    for review in root.findall('Review'):
        for sentences in review.findall('sentences'):
            for sentence in sentences.findall('sentence'):
                text = sentence.find('text').text  # 句子
                if not sentence.findall('Opinions'):  # 删除没有aspect的句子
                    continue
                for opinions in sentence.findall('Opinions'):
                    for opinion in opinions.findall('Opinion'):  # 遍历所有的aspect
                        if opinion.get('target') == 'NULL':
                            continue
                        polarity = opinion.get('polarity').strip()
                        start = opinion.get('from')
                        end = opinion.get('to')
                        mask_sentence_list.append((text[:int(start)] + ' $T$ ' + text[int(end):]).strip())
                        aspect_list.append(opinion.get('target').strip())
                        label_list.append(sentiment_dict[polarity])
                        start_list.append(start)
                        end_list.append(end)
    fout = open(outfile, 'w', encoding='utf-8')
    data = []
    for (i, j, k, l, m) in zip(mask_sentence_list, aspect_list, label_list, start_list, end_list):
        if (i, j, k, l, m) not in data:
            data.append((i, j, k, l, m))
            fout.write(i + '\n' + j + '\n' + k + '\n')
    fout.close()


def xml2txt():
    # xml_parse_14('./datasets/xml_v2/SemEval2014/Laptops_Train.xml', './datasets/xml2txt/lap14_train.txt')
    # xml_parse_14('./datasets/xml_v2/SemEval2014/Laptops_Test.xml', './datasets/xml2txt/lap14_test.txt')
    # xml_parse_14('./datasets/xml_v2/SemEval2014/Restaurants_Train.xml', './datasets/xml2txt/rest14_train.txt')
    # xml_parse_14('./datasets/xml_v2/SemEval2014/Restaurants_Test.xml', './datasets/xml2txt/rest14_test.txt')
    # xml_parse_1516('./datasets/xml_v2/SemEval2015/Restaurants_Train.xml', './datasets/xml2txt/rest15_train.txt')
    # xml_parse_1516('./datasets/xml_v2/SemEval2015/Restaurants_Test.xml', './datasets/xml2txt/rest15_test.txt')
    # xml_parse_1516('./datasets/xml_v2/SemEval2016/Restaurants_Train.xml', './datasets/xml2txt/rest16_train.txt')
    # xml_parse_1516('./datasets/xml_v2/SemEval2016/Restaurants_Test.xml', './datasets/xml2txt/rest16_test.txt')
    return


def show_diff(infile, goldfile):
    fin1 = open(infile, 'r', encoding='utf-8')
    fin2 = open(goldfile, 'r', encoding='utf-8')
    lines1 = fin1.readlines()
    lines2 = fin2.readlines()
    fin1.close()
    fin2.close()

    mask_sent1, mask_sent2 = [], []
    aspect1, aspect2 = [], []
    label1, label2 = [], []

    for i in range(0, len(lines1), 3):
        mask_sent = lines1[i].strip()
        aspect = lines1[i + 1].strip()
        label = lines1[i + 2].strip()
        aspect1.append(aspect.lower())
        label1.append(label)
        mask_sent1.append(mask_sent.lower())

    for i in range(0, len(lines2), 3):
        mask_sent = lines2[i].strip()
        aspect = lines2[i + 1].strip()
        label = lines2[i + 2].strip()
        aspect2.append(aspect.lower())
        label2.append(label)
        mask_sent2.append(mask_sent.lower())
    count, count2 = 1, 1
    if len(lines1) == len(lines2):
        print('number is same')
        print('-' * 10)
        print('diff in aspect, check the line in {}'.format(goldfile))
        for i, j in zip(aspect1, aspect2):
            if i != j:
                print(count * 3 - 1)
            count += 1
        print('-' * 10)
        print('diff in label, check the line in {}'.format(goldfile))
        for i, j in zip(label1, label2):
            if i != j:
                print(count * 3 - 1)
            count += 1
        print('-' * 10)
    else:
        if len(lines1) < len(lines2):
            print('ours is smaller')
            mask_sent1 += (len(lines2) - len(lines1)) * ['a']
        else:
            print('ours is larger')
            mask_sent2 += (len(lines1) - len(lines2)) * ['a']
        last_diff = 0
        for i, j in zip(aspect1, aspect2):
            if i != j:
                print(count * 3)
                if (count * 3) - last_diff == 3:
                    print('maybe miss data in line{}, please check and revise {} and run it again.'.format(last_diff,
                                                                                                           infile))
                last_diff = count * 3
            count += 1


def compare():
    # show_diff('./datasets/xml2txt/lap14_train.txt', './datasets/golden/lap14_train.txt')
    # show_diff('./datasets/xml2txt/lap14_test.txt', './datasets/golden/lap14_test.txt')
    # show_diff('./datasets/xml2txt/rest14_train.txt', './datasets/golden/rest14_train.txt')
    # show_diff('./datasets/xml2txt/rest14_test.txt', './datasets/golden/rest14_test.txt')
    # show_diff('./datasets/xml2txt/rest15_train.txt', './datasets/golden/rest15_train.txt')
    # show_diff('./datasets/xml2txt/rest15_test.txt', './datasets/golden/rest15_test.txt')
    # show_diff('./datasets/xml2txt/rest16_train.txt', './datasets/golden/rest16_train.txt')
    # show_diff('./datasets/xml2txt/rest16_test.txt', './datasets/golden/rest16_test.txt')
    return


def tokenizer(infile, outfile, parser='spacy'):
    nlp = None
    if parser == 'spacy':
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_md")
    elif parser == 'stanza':
        nlp = stanza.Pipeline('en', processors='tokenize', download_method="None")
    else:
        print('can not find {} parser'.format(parser))

    fin = open(infile, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    fout = open(outfile, 'w', encoding='utf-8')
    for i in tqdm(range(0, len(lines), 3)):
        mask_text = lines[i].strip()
        mask_text = re.sub(r'\s\s', ' ', mask_text)
        mask_text = re.sub(r'\$T\$', 'TARGET', mask_text)
        doc = nlp(mask_text)
        words = []
        if parser == 'spacy':
            for token in doc:
                words.append(token.text)
        elif parser == 'stanza':
            for sentence in doc.sentences:
                for word in sentence.words:
                    words.append(word.text)
        mask_text_tokenize = ' '.join(words)
        print(mask_text_tokenize)
        mask_text_tokenize = re.sub(r'TARGET', '$T$', mask_text_tokenize)
        fout.write(mask_text_tokenize + '\n')
        aspect = lines[i + 1].strip()
        doc2 = nlp(aspect)
        words2 = []
        if parser == 'spacy':
            for token in doc2:
                words2.append(token.text)
        elif parser == 'stanza':
            for sentence in doc2.sentences:
                for word in sentence.words:
                    words2.append(word.text)
        aspect = ' '.join(words2)
        fout.write(aspect + '\n')
        fout.write(lines[i + 2].strip() + '\n')
    fout.close()


def write_data(parser='spacy'):
    tokenizer('./datasets/xml2txt/lap14_train.txt', './datasets/{}/lap14_train.txt'.format(parser), parser)
    tokenizer('./datasets/xml2txt/lap14_test.txt', './datasets/{}/lap14_test.txt'.format(parser), parser)
    tokenizer('./datasets/xml2txt/rest14_train.txt', './datasets/{}/rest14_train.txt'.format(parser), parser)
    tokenizer('./datasets/xml2txt/rest14_test.txt', './datasets/{}/rest14_test.txt'.format(parser), parser)
    tokenizer('./datasets/xml2txt/rest15_train.txt', './datasets/{}/rest15_train.txt'.format(parser), parser)
    tokenizer('./datasets/xml2txt/rest15_test.txt', './datasets/{}/rest15_test.txt'.format(parser), parser)
    tokenizer('./datasets/xml2txt/rest16_train.txt', './datasets/{}/rest16_train.txt'.format(parser), parser)
    tokenizer('./datasets/xml2txt/rest16_test.txt', './datasets/{}/rest16_test.txt'.format(parser), parser)


def show_parser_diff(infile1, infile2):
    fin1 = open(infile1, 'r', encoding='utf-8')
    lines1 = fin1.readlines()
    fin1.close()
    fin2 = open(infile2, 'r', encoding='utf-8')
    lines2 = fin2.readlines()
    fin2.close()

    mask_sent1, mask_sent2 = [], []
    aspect1, aspect2 = [], []
    label1, label2 = [], []
    for i in range(0, len(lines1), 3):
        mask_sent = lines1[i].strip()
        aspect = lines1[i + 1].strip()
        label = lines1[i + 2].strip()
        mask_sent1.append(mask_sent)
        aspect1.append(aspect)
        label1.append(label)
    for i in range(0, len(lines2), 3):
        mask_sent = lines2[i].strip()
        aspect = lines2[i + 1].strip()
        label = lines2[i + 2].strip()
        mask_sent2.append(mask_sent)
        aspect2.append(aspect)
        label2.append(label)
    for s1, s2, a1, a2, l1, l2 in zip(mask_sent1, mask_sent2, aspect1, aspect2, label1, label2):
        if (s1, a1, l1) != (s2, a2, l2):
            print(s1)
            print(s2)
            print(a1)
            print(a2)
            print(l1)
            print(l2)


def compare_parser():
    show_parser_diff('./datasets/spacy/lap14_train.txt', './datasets/stanza/lap14_train.txt')
    show_parser_diff('./datasets/spacy/lap14_test.txt', './datasets/stanza/lap14_test.txt')
    show_parser_diff('./datasets/spacy/rest14_train.txt', './datasets/stanza/rest14_train.txt')
    show_parser_diff('./datasets/spacy/rest14_test.txt', './datasets/stanza/rest14_test.txt')
    show_parser_diff('./datasets/spacy/rest15_train.txt', './datasets/stanza/rest15_train.txt')
    show_parser_diff('./datasets/spacy/rest15_test.txt', './datasets/stanza/rest15_test.txt')
    show_parser_diff('./datasets/spacy/rest16_train.txt', './datasets/stanza/rest16_train.txt')
    show_parser_diff('./datasets/spacy/rest16_test.txt', './datasets/stanza/rest16_test.txt')


def lookup(glove, text, case):
    if case == 'lowercase':
        text = text.lower()
    new_text = ' '
    for word in text.strip().split():
        if word.lower() in glove:
            new_text += word + ' '
        else:
            if '-' in word:
                new_text += re.sub(r'-', ' - ', word) + ' '
            elif '/' in word:
                new_text += re.sub(r'/', ' / ', word) + ' '
            elif '\'' in word:
                new_text += re.sub(r'\'', " ' ", word) + ' '
            elif '$' in word:
                new_text += re.sub(r'\$', ' $ ', word) + ' '
            else:
                new_text += word + ' '
    return new_text


def glove_tokenizer(glove, infile, outfile, case):
    fin = open(infile, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    fout = open(outfile, 'w', encoding='utf-8')
    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = lines[i].strip().partition('$T$')
        aspect = lines[i + 1].strip()
        text1 = lookup(glove, text_left, case)
        text2 = lookup(glove, aspect, case)
        text3 = lookup(glove, text_right, case)
        # print(text1 + text2 + text3)
        fout.write((text1.strip() + ' $T$ ' + text3.strip()).strip() + '\n')
        fout.write(text2.strip() + '\n')
        fout.write(lines[i + 2].strip() + '\n')


def write_data_glove(case='lowercase'):
    fin = open('./glove_vocab.txt', 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    vocab = []
    for line in lines:
        vocab.append(line.strip())
    glove_tokenizer(vocab, './datasets/regex_data/lap14_train.txt', './datasets/final_{}/lap14_train.txt'.format(case),
                    case)
    glove_tokenizer(vocab, './datasets/regex_data/lap14_test.txt', './datasets/final_{}/lap14_test.txt'.format(case),
                    case)
    glove_tokenizer(vocab, './datasets/regex_data/rest14_train.txt',
                    './datasets/final_{}/rest14_train.txt'.format(case), case)
    glove_tokenizer(vocab, './datasets/regex_data/rest14_test.txt', './datasets/final_{}/rest14_test.txt'.format(case),
                    case)
    glove_tokenizer(vocab, './datasets/regex_data/rest15_train.txt',
                    './datasets/final_{}/rest15_train.txt'.format(case), case)
    glove_tokenizer(vocab, './datasets/regex_data/rest15_test.txt', './datasets/final_{}/rest15_test.txt'.format(case),
                    case)
    glove_tokenizer(vocab, './datasets/regex_data/rest16_train.txt',
                    './datasets/final_{}/rest16_train.txt'.format(case), case)
    glove_tokenizer(vocab, './datasets/regex_data/rest16_test.txt', './datasets/final_{}/rest16_test.txt'.format(case),
                    case)


def read_data(glove, infile):
    fin = open(infile, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    vocab = set()
    error_word = set()
    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = lines[i].strip().partition('$T$')
        aspect = lines[i + 1].strip()
        text = text_left + aspect + text_right
        for word in text.lower().split():
            vocab.add(word)
            if word not in glove:
                error_word.add(word)
    print(error_word)
    print('-' * 20)
    print('total: {}'.format(len(vocab)))
    print(' unk : {}'.format(len(error_word)))
    print('{}/{}={}%'.format(len(error_word), len(vocab), round(len(error_word) / len(vocab) * 100, 4)))


def unk_token():
    fin = open('./glove_vocab.txt', 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    vocab = []
    for line in lines:
        vocab.append(line.strip())
    print(len(vocab))
    read_data(vocab, './datasets/final_lowercase/lap14_train.txt')
    read_data(vocab, './datasets/final_lowercase/lap14_test.txt')
    read_data(vocab, './datasets/final_lowercase/rest14_train.txt')
    read_data(vocab, './datasets/final_lowercase/rest14_test.txt')
    read_data(vocab, './datasets/final_lowercase/rest15_train.txt')
    read_data(vocab, './datasets/final_lowercase/rest15_test.txt')
    read_data(vocab, './datasets/final_lowercase/rest16_train.txt')
    read_data(vocab, './datasets/final_lowercase/rest16_test.txt')


def lower(infile, outfile):
    fin = open(infile, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    fout = open(outfile, 'w', encoding='utf-8')
    for i in range(0, len(lines), 3):
        text = lines[i].strip().split()
        text = ' '.join(text)
        aspect = lines[i + 1].strip().split()
        label = lines[i + 2].strip()
        aspect = ' '.join(aspect)
        new_text = ''
        for word in text.strip().split():
            if word not in ['-LRB-', '-RRB-', '-LSB-', '-RSB-', '$T$']:
                word = word.lower()
            new_text += word + ' '
        fout.write(new_text.strip() + '\n')
        new_aspect = ''
        for word in aspect.strip().split():
            if word not in ['-LRB-', '-RRB-', '-LSB-', '-RSB-', '$T$']:
                word = word.lower()
            new_aspect += word + ' '
        fout.write(new_aspect.strip() + '\n')
        fout.write(label + '\n')
    fout.close()


def write_lower_data():
    lower("./datasets/golden/lap14_train.txt", './one_sentence_one_aspect/golden/lap14_train.txt')
    lower("./datasets/golden/lap14_test.txt", './one_sentence_one_aspect/golden/lap14_test.txt')
    lower("./datasets/golden/rest14_train.txt", './one_sentence_one_aspect/golden/rest14_train.txt')
    lower("./datasets/golden/rest14_test.txt", './one_sentence_one_aspect/golden/rest14_test.txt')
    lower("./datasets/golden/rest15_train.txt", './one_sentence_one_aspect/golden/rest15_train.txt')
    lower("./datasets/golden/rest15_test.txt", './one_sentence_one_aspect/golden/rest15_test.txt')
    lower("./datasets/golden/rest16_train.txt", './one_sentence_one_aspect/golden/rest16_train.txt')
    lower("./datasets/golden/rest16_test.txt", './one_sentence_one_aspect/golden/rest16_test.txt')


def merge_sent(infile, outfile):
    fin = open(infile, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    fout = open(outfile, 'w', encoding='utf-8')
    data = []
    for i in tqdm(range(0, len(lines), 3)):
        left_text, _, right_text = lines[i].strip().partition('$T$')
        aspect = lines[i + 1].strip()
        label = lines[i + 2].strip()
        text = left_text + aspect + right_text
        aspect_start = len(left_text.split())
        aspect_end = len(left_text.split()) + len(aspect.split()) - 1
        data.append([text, aspect, aspect_start, aspect_end, label])

    last_sent = data[0][0]
    new_data = [data[0]]
    merge_data = []
    for i in data[1:]:
        if i[0] == last_sent:
            new_data.append(i)
            if i == data[-1]:
                merge_data.append(new_data)
        else:
            merge_data.append(new_data)
            new_data = [i]
            last_sent = i[0]
            if i == data[-1]:
                merge_data.append(new_data)
    for merge_list in merge_data:
        fout.write(merge_list[0][0] + '\n')
        aspect_temp, label_temp = '', ''
        for j in merge_list:
            aspect_info = '{}_{}_{} [SEP] '.format(j[1], str(j[2]), str(j[3]))
            aspect_temp += aspect_info
            label_info = j[4] + ' '
            label_temp += label_info
        fout.write(aspect_temp + '\n' + label_temp + '\n')


def merge(case='golden'):
    merge_sent("./one_sentence_one_aspect/{}/lap14_train.txt".format(case),
               './one_sentence_all_aspect/{}/lap14_train.txt'.format(case))
    merge_sent("./one_sentence_one_aspect/{}/lap14_test.txt".format(case),
               './one_sentence_all_aspect/{}/lap14_test.txt'.format(case))
    merge_sent("./one_sentence_one_aspect/{}/rest14_train.txt".format(case),
               './one_sentence_all_aspect/{}/rest14_train.txt'.format(case))
    merge_sent("./one_sentence_one_aspect/{}/rest14_test.txt".format(case),
               './one_sentence_all_aspect/{}/rest14_test.txt'.format(case))
    merge_sent("./one_sentence_one_aspect/{}/rest15_train.txt".format(case),
               './one_sentence_all_aspect/{}/rest15_train.txt'.format(case))
    merge_sent("./one_sentence_one_aspect/{}/rest15_test.txt".format(case),
               './one_sentence_all_aspect/{}/rest15_test.txt'.format(case))
    merge_sent("./one_sentence_one_aspect/{}/rest16_train.txt".format(case),
               './one_sentence_all_aspect/{}/rest16_train.txt'.format(case))
    merge_sent("./one_sentence_one_aspect/{}/rest16_test.txt".format(case),
               './one_sentence_all_aspect/{}/rest16_test.txt'.format(case))


def compare_sent(infile):
    print('-' * 10 + infile + '-' * 10)
    fin = open(infile, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    all_text = []
    for i in range(0, len(lines), 3):
        text = lines[i].strip()
        all_text.append(text.split())

    last_text = all_text[0]
    for i in all_text[1:]:
        if len(set(last_text).difference(set(i))) <= 2 and abs(len(last_text) - len(i)) <= 5:
            print(' '.join(last_text))
            print(' '.join(i))
            print()
        else:
            last_text = i


def merge_check(case='lowercase'):
    # compare_sent('./one_sentence_all_aspect/final_{}/lap14_train.txt'.format(case))
    # compare_sent('./one_sentence_all_aspect/final_{}/lap14_test.txt'.format(case))
    # compare_sent('./one_sentence_all_aspect/final_{}/rest14_train.txt'.format(case))
    # compare_sent('./one_sentence_all_aspect/final_{}/rest14_test.txt'.format(case))
    # compare_sent('./one_sentence_all_aspect/final_{}/rest15_train.txt'.format(case))
    # compare_sent('./one_sentence_all_aspect/final_{}/rest15_test.txt'.format(case))
    # compare_sent('./one_sentence_all_aspect/final_{}/rest16_train.txt'.format(case))
    # compare_sent('./one_sentence_all_aspect/final_{}/rest16_test.txt'.format(case))

    compare_sent('./one_sentence_all_aspect/{}/lap14_train.txt'.format(case))
    compare_sent('./one_sentence_all_aspect/{}/lap14_test.txt'.format(case))
    compare_sent('./one_sentence_all_aspect/{}/rest14_train.txt'.format(case))
    compare_sent('./one_sentence_all_aspect/{}/rest14_test.txt'.format(case))
    compare_sent('./one_sentence_all_aspect/{}/rest15_train.txt'.format(case))
    compare_sent('./one_sentence_all_aspect/{}/rest15_test.txt'.format(case))
    compare_sent('./one_sentence_all_aspect/{}/rest16_train.txt'.format(case))
    compare_sent('./one_sentence_all_aspect/{}/rest16_test.txt'.format(case))


def token_glove(glove, infile, outfile):
    fin = open(infile, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    for i in range(0, len(lines), 3):
        text_left, _, text_right = lines[i].strip().partition('$T$')
        aspect = lines[i + 1].strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        for word in text.split():
            if word not in ['-LRB-', '-RRB-']:
                word = word.lower()
            if word not in glove and not word.isalpha():
                print('line:{} word:{}'.format(i, word))


def tokenizer_again():
    fin = open('./glove_vocab.txt', 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    vocab = []
    for line in lines:
        vocab.append(line.strip())
    # token_glove(vocab, './datasets/golden/lap14_train.txt', './one_sentence_one_aspect/lap14_train.txt')
    # token_glove(vocab, './datasets/golden/lap14_test.txt', './one_sentence_one_aspect/lap14_test.txt')
    # token_glove(vocab, './datasets/golden/rest14_train.txt', './one_sentence_one_aspect/rest14_train.txt')
    # token_glove(vocab, './datasets/golden/rest14_test.txt', './one_sentence_one_aspect/rest14_test.txt')
    # token_glove(vocab, './datasets/golden/rest15_train.txt', './one_sentence_one_aspect/rest15_train.txt')
    # token_glove(vocab, './datasets/golden/rest15_test.txt', './one_sentence_one_aspect/rest15_test.txt')
    # token_glove(vocab, './datasets/golden/rest16_train.txt', './one_sentence_one_aspect/rest16_train.txt')
    # token_glove(vocab, './datasets/golden/rest16_test.txt', './one_sentence_one_aspect/rest16_test.txt')


if __name__ == '__main__':
    # xml2txt()
    # compare()
    # write_data('stanza')
    # write_data('spacy')
    # compare_parser()
    # write_data_glove('uppercase')
    # unk_token()
    """ignore"""

    # write_lower_data()
    merge('golden')  # 合并aspect
    merge_check('golden')  # 保证一致
    # tokenizer_again()
