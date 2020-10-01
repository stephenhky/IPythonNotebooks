
# reference: https://towardsdatascience.com/word-embeddings-in-2020-review-with-code-examples-11eb39a1ee6d

from simplerepresentations import RepresentationModel


class SimpleRepresentationsSentenceEncoder:
    def __init__(self):
        self.representation_model = RepresentationModel(
            model_type='bert',
            model_name='bert-base-uncased',
            batch_size=5,
            max_seq_length=48,
            combination_method='cat',
            last_hidden_to_use=4
        )

    def encode_sentences(self, sentences):
        all_sentences_representations, all_token_representations = \
            self.representation_model(text_a=sentences)
        return all_sentences_representations, all_token_representations
