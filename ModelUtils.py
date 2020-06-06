import numpy as np
import torch
from torch import nn

from parlai.core.dict import DictionaryAgent


def universal_sentence_embedding(sentences, mask, sqrt=True):
    """
    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings
    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    # need to mask out the padded chars
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


def load_embedding(dictionary: DictionaryAgent, path: str) -> nn.Embedding:
    """
        获取embedding
        :param dictionary: 目标字典
        :param path: numpy文件路径
        :return: Embedding对象，大小为 词表 x 维度
        """
    weights = np.load(path)
    emb_matrix = torch.from_numpy(weights)
    embedding = nn.Embedding(len(dictionary.keys(), 300)).from_pretrained(emb_matrix.to(torch.float32))
    return embedding
