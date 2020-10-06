
# reference: https://towardsdatascience.com/word-embeddings-in-2020-review-with-code-examples-11eb39a1ee6d

import argparse
from itertools import product

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel


class WrappedBERTEncoder:
    def __init__(self, model=None, tokenizer=None):
        if model is None:
            self.model = BertModel.from_pretrained('bert-base-uncased',
                                                   output_hidden_states=True)
        elif isinstance(model, BertModel):
            self.model = model
        elif isinstance(model, str):
            self.model = BertModel.from_pretrained(model, output_hidden_states=True)
        else:
            raise Exception('Invalid model.')

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif isinstance(tokenizer, BertTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case=True)
        else:
            raise Exception('Invalid tokenizer.')

    def encode_sentences(self, sentences):
        input_ids = []
        # attention_masks = []
        tokenized_texts = []

        for sentence in sentences:
            marked_text = '[CLS]' + sentence + '[SEP]'

            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                truncation=True,
                max_length=48,
                pad_to_max_length=True,
                return_tensors='pt'
            )

            tokenized_texts.append(self.tokenizer.tokenize(marked_text))
            input_ids.append(encoded_dict['input_ids'])

        input_ids = torch.cat(input_ids, dim=0)
        segments_id = torch.LongTensor(np.array(input_ids > 0))

        with torch.no_grad():
            _, sentences_embeddings, hidden_state = self.model(input_ids, segments_id)

        token_embeddings = torch.stack(hidden_state, dim=0)
        token_embeddings = token_embeddings.permute(1, 2, 0, 3)  # swap dimensions to [sentence, tokens, hidden layers, features]
        processed_embeddings = token_embeddings[:, :, 9:, :]   # we want last 4 layers only

        embeddings = torch.reshape(processed_embeddings, (len(sentences), 48, -1))
        embeddings = embeddings.detach().numpy()

        return sentences_embeddings, embeddings, tokenized_texts


def get_argparser():
    argparser = argparse.ArgumentParser(description='Contextual similarity with BERT.')
    argparser.add_argument('--model', default='bert-base-uncased', help='BERT model (default: "bert-case-uncase")')
    return argparser


if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()
    modelname = args.model

    print('Loading BERT model...')
    encoder = WrappedBERTEncoder(model=modelname, tokenizer=modelname)

    print('BERT Contextual Similarities')
    sentence1 = input('Sentence 1? ')
    sentence2 = input('Sentence 2? ')

    sentences_embeddings, embeddings, tokenized_texts = encoder.encode_sentences([sentence1, sentence2])

    # sentence similarity
    sentence_similarity = 1-cosine_distance(sentences_embeddings[0], sentences_embeddings[1])
    print('Cosine similarity between two sentences: {}'.format(sentence_similarity))

    # token similarities
    simmatrix = np.zeros((len(tokenized_texts[0]), len(tokenized_texts[1])))
    for i, j in product(range(len(tokenized_texts[0])), range(len(tokenized_texts[1]))):
        simmatrix[i, j] = 1 - cosine_distance(embeddings[0, i], embeddings[1, j])
    simdf = pd.DataFrame(simmatrix)
    simdf.columns = tokenized_texts[1]
    simdf.index = tokenized_texts[0]
    print(simdf)
