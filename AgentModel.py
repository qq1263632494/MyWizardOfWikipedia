from typing import List, Tuple, Any, Dict

import numpy as np
import torch
from torch import nn

from DataUtils import WizardOfWikipediaExample
from FeatureUtils import Tokenizer
from ModelUtils import universal_sentence_embedding
from parlai.agents.transformer.modules import TransformerEncoder, TransformerDecoder


class ContextKnowledgeEncoder(nn.Module):
    def __init__(self, transformer: TransformerEncoder):
        super().__init__()
        self.embeddings = transformer.embeddings
        self.embed_dim = transformer.embeddings.embedding_dim
        self.transformer = transformer

    def forward(self, src_tokens, know_tokens, cs_ids, use_cs_ids):
        context_encoded, context_mask = self.transformer(src_tokens)
        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded, know_mask = self.transformer(know_flat)
        context_use = universal_sentence_embedding(context_encoded, context_mask)
        know_use = universal_sentence_embedding(know_encoded, know_mask)
        know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim)
        context_use /= np.sqrt(self.embed_dim)
        know_use /= np.sqrt(self.embed_dim)
        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        if not use_cs_ids:
            _, cs_ids = ck_attn.max(1)
        cs_offsets = torch.arange(N, device=cs_ids.device) * K + cs_ids
        cs_encoded = know_encoded[cs_offsets]
        cs_mask = know_mask[cs_offsets]
        full_enc = torch.cat([cs_encoded, context_encoded], dim=1)
        full_mask = torch.cat([cs_mask, context_mask], dim=1)
        return full_enc, full_mask, ck_attn


class ContextKnowledgeDecoder(nn.Module):
    def __init__(self, transformer: TransformerDecoder):
        super().__init__()
        self.transformer = transformer

    def forward(self, input, encoder_state, incr_state=None):
        # our CK Encoder returns an extra output which the Transformer decoder
        # doesn't expect (the knowledge selection mask). Just chop it off
        encoder_output, encoder_mask, _ = encoder_state
        return self.transformer(input, (encoder_output, encoder_mask), incr_state)


class ModelOption:
    def __init__(self,
                 n_heads=2,
                 n_layers=5,
                 ffn_size=512,
                 dropout=0.2,
                 n_positions=128):
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_size = ffn_size
        self.dropout = dropout
        self.n_positions = n_positions


class TransformerMemNet(nn.Module):
    def __init__(self, opt: ModelOption,
                 tokenizer: Tokenizer,
                 embedding: nn.Embedding):
        super(TransformerMemNet, self).__init__()
        self.encoder = TransformerEncoder(n_heads=opt.n_heads,
                                          n_layers=opt.n_layers,
                                          embedding_size=embedding.weight.shape[1],
                                          vocabulary_size=embedding.weight.shape[0],
                                          ffn_size=opt.ffn_size,
                                          embedding=embedding,
                                          dropout=opt.dropout,
                                          padding_idx=tokenizer.dictionary[tokenizer.dictionary.null_token],
                                          n_positions=opt.n_positions, reduction_type='none')
        self.decoder = TransformerDecoder(n_heads=opt.n_heads,
                                          n_layers=opt.n_layers,
                                          embedding_size=embedding.weight.shape[1],
                                          ffn_size=opt.ffn_size,
                                          vocabulary_size=embedding.weight.shape[0],
                                          embedding=embedding,
                                          dropout=opt.dropout,
                                          n_positions=opt.n_positions,
                                          padding_idx=tokenizer.dictionary[tokenizer.dictionary.null_token])
        self.encoder = ContextKnowledgeEncoder(self.encoder)
        self.decoder = ContextKnowledgeDecoder(self.decoder)
        self.linear = nn.Linear(300, embedding.weight.shape[0])

    def forward(self,
                utterances: torch.LongTensor,
                knowledge_pools: torch.LongTensor,
                response_inputs: torch.LongTensor,
                choose_indexes: torch.LongTensor = None) -> Tuple[torch.Tensor, Any]:
        if choose_indexes is not None:
            out = self.encoder(utterances, knowledge_pools, choose_indexes, True)
        else:
            out = self.encoder(utterances, knowledge_pools, None, False)
        predict_tokens_vector, _ = self.decoder(response_inputs, out, None)
        predict_tokens = self.linear(predict_tokens_vector)
        return predict_tokens, out[2]


class TransformerMemNetAgent:
    def __init__(self, opt: ModelOption, tokenizer: Tokenizer, embedding: nn.Embedding):
        self.tokenizer = tokenizer
        self.model = TransformerMemNet(opt, tokenizer, embedding)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_index, reduction='mean')

    def compute_loss(self, examples: List[WizardOfWikipediaExample]) -> Tuple[torch.Tensor, torch.Tensor]:
        utterances, knowledge_pools, response_inputs, response_outputs = self.tokenizer.tokenize_example_batch(examples)
        choose_indexes = []
        for example in examples:
            choose_indexes.append(example.choosen_index)
        choose_indexes = torch.LongTensor(choose_indexes)
        predict_probabilities, utterance_knowledge_attentions = self.model(utterances,
                                                                           knowledge_pools,
                                                                           response_inputs,
                                                                           choose_indexes)
        return self.loss_func(predict_probabilities.view(-1, predict_probabilities.size(-1)),
                              response_outputs.view(-1)), self.loss_func(utterance_knowledge_attentions,
                                                                         choose_indexes)

    def compute_train_batch_loss(self, data: Dict, alpha: float):
        choose_indexes = data['choose_indexes']
        response_outputs = data['response_outputs']
        del data['response_outputs']
        predict_probabilities, utterance_knowledge_attentions = self.model(**data)
        tokens_loss = self.loss_func(predict_probabilities.view(-1, predict_probabilities.size(-1)),
                                     response_outputs.view(-1))
        knowledge_choose_loss = self.loss_func(utterance_knowledge_attentions,
                                               choose_indexes)
        return tokens_loss * alpha + knowledge_choose_loss * (1 - alpha)

    def generate_response(self, example: WizardOfWikipediaExample, device='cuda:0'):
        utterances, knowledge_pools, response_inputs, response_outputs = self.tokenizer.tokenize_example_batch(
            [example])

        utterances = utterances.to(device)
        knowledge_pools = knowledge_pools.to(device)
        response_inputs = response_inputs.to(device)

        predict_probabilities, utterance_knowledge_attentions = self.model(utterances, knowledge_pools,
                                                                           response_inputs)
        predict_token_sequence = torch.argmax(predict_probabilities, dim=2).squeeze(0).cpu().numpy().tolist()
        choose_index = torch.argmax(utterance_knowledge_attentions, dim=1).squeeze(0).cpu().numpy().tolist()

        real_predict_token_sequence = []
        for token in predict_token_sequence:
            if token != self.tokenizer.end_token_index:
                real_predict_token_sequence.append(token)

        predict_sentence = self.tokenizer.dictionary.vec2txt(real_predict_token_sequence)
        print('[UTTERANCE]')
        print(example.utterance)
        print('[REAL GOLDEN SENTENCE]')
        print(example.golden_sentence)
        print('[REAL RESPONSE]')
        print(example.response)
        print('[PREDICT GOLDEN SENTENCE]')
        print(example.knowledge_pool[choose_index])
        print('[PREDICT UTTERANCE]')
        print(predict_sentence)
        return predict_sentence, choose_index
