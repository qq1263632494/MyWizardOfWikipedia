# 生成特征所使用的工具
from typing import List, Tuple, Union, Dict

import torch

from DataUtils import WizardOfWikipediaExample
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt


class Tokenizer:
    def __init__(self,
                 dictionary: DictionaryAgent,
                 pad_to_max=True,
                 max_length=128):
        """
                分词并生成对应向量的工具类
                :param dictionary: 字典工具对象
                :param pad_to_max:是否pad到指定长度
                :param max_length:指定的长度
                """
        self.dictionary = dictionary
        self.pad_to_max = pad_to_max
        self.max_length = max_length
        self.start_token_index = dictionary[dictionary.start_token]
        self.end_token_index = dictionary[dictionary.end_token]
        self.pad_token_index = dictionary[dictionary.null_token]

    def tokenize(self, sentence: str, return_tensor=False):
        """
            对一个句子进行编码
            :param sentence: 目标句子
            :param return_tensor:是否返回Tensor
        """
        tokens_ids = self.dictionary.txt2vec(sentence)
        if self.pad_to_max:
            real_length = len(tokens_ids)
            for i in range(real_length, self.max_length):
                tokens_ids.append(self.dictionary[self.dictionary.null_token])
            tokens_ids = tokens_ids[0:self.max_length]
        if return_tensor:
            return torch.LongTensor(tokens_ids)
        else:
            return tokens_ids

    def tokenize_response(self, sentence: str) -> Tuple[List[int], List[int]]:
        response_tokens_indexes = self.dictionary.txt2vec(sentence)
        response_input_indexes = [self.start_token_index] + response_tokens_indexes
        length = len(response_input_indexes)
        for i in range(length, self.max_length):
            response_input_indexes.append(self.pad_token_index)
        length = len(response_tokens_indexes)
        if length > self.max_length - 1:
            response_output_indexes = response_tokens_indexes[0:self.max_length - 1] + [self.end_token_index]
        else:
            response_output_indexes = response_tokens_indexes + [self.end_token_index] + [self.pad_token_index for i in
                                                                                          range(length + 1,
                                                                                                self.max_length)]
        return response_input_indexes[0:self.max_length], response_output_indexes[0:self.max_length]

    def tokenize_example(self, example: WizardOfWikipediaExample) \
            -> Tuple[
                Union[torch.LongTensor, List[int]], List[Union[torch.LongTensor, List[int]]], List[int], List[int]]:
        """
            对一个WizardOfWikipediaExample对象进行编码
            :param example: 目标WizardOfWikipediaExample对象
            :return 四个返回值，
            第一个为utterance的编码，
            第二个是所有知识的编码，
            第三个是response_input的编码，
            第四个是response_output的编码
        """
        src_tokens = self.tokenize(example.utterance)
        know_tokens = []
        response_input_indexes, response_output_indexes = self.tokenize_response(example.response)
        for i in range(0, 32):
            know_tokens.append(self.tokenize(example.knowledge_pool[i]))
        return src_tokens, know_tokens, response_input_indexes, response_output_indexes

    def tokenize_example_batch(self, example_list: List[WizardOfWikipediaExample]) \
            -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
                                        对一组WizardOfWikipediaExample对象进行编码
                                        :param example_list: 若干WizardOfWikipediaExample对象
                                        :return 三个返回值，第一个为utterance的编码,形状为 批大小 x 序列长度，
                                        第二个是所有知识的编码，形状为 批大小 x 32 x 序列长度，
                                        第三个是response的编码，形状为 批大小 x 序列长度，
                                        """
        src_tokens_list = []
        know_tokens_list = []
        response_input_list = []
        response_output_list = []
        for example in example_list:
            src_tokens, know_tokens, response_input_indexes, response_output_indexes = self.tokenize_example(example)
            src_tokens_list.append(src_tokens)
            know_tokens_list.append(know_tokens)
            response_input_list.append(response_input_indexes)
            response_output_list.append(response_output_indexes)
        return torch.LongTensor(src_tokens_list), \
               torch.LongTensor(know_tokens_list), \
               torch.LongTensor(response_input_list), \
               torch.LongTensor(response_output_list)

    def tokenize_example_as_train_data(self, example: WizardOfWikipediaExample) -> Dict:
        utterances, knowledge_pools, response_inputs, response_outputs = self.tokenize_example(example)
        choose_indexes = example.choosen_index
        return {
            'utterances': torch.LongTensor(utterances),
            'knowledge_pools': torch.LongTensor(knowledge_pools),
            'response_inputs': torch.LongTensor(response_inputs),
            'response_outputs': torch.LongTensor(response_outputs),
            'choose_indexes': choose_indexes
        }


def get_dictionary(PATH: str) -> DictionaryAgent:
    """
                    读取字典
                    :param PATH: 字典工具目录
                    :return 读取的字典
                    """
    opt = Opt()
    dictionary = DictionaryAgent(opt=opt)
    dictionary.load(PATH)
    return dictionary
