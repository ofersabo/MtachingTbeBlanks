from typing import Dict
from typing import List
import json
import logging

from overrides import overrides
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import ListField, IndexField, MetadataField, Field

head_start_token = '[unused1]'  # fixme check if this indeed the token they used in the paper
head_end_token = '[unused2]'  # fixme check if this indeed the token they used in the paper
tail_start_token = '[unused3]'
tail_end_token = '[unused4]'

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def find_closest_distance_between_entities(head_start_location, head_end_location, tail_start_location,
                                           tail_end_location):
    min_distance = 99999
    for i, x in enumerate(head_start_location):
        for j, y in enumerate(tail_start_location):
            if abs(x - y) < min_distance:
                min_distance = abs(x - y)
                h_start, h_end, t_start, t_end = x, head_end_location[i], y, tail_end_location[j]

    return h_start, h_end, t_start, t_end


@DatasetReader.register("only_entities_reader")
class onlyEntitiesDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.spacy_splitter = SpacyWordSplitter(keep_spacy_tokens=True)
        self.TRAIN_DATA = "meta_train"
        self.TEST_DATA = "meta_test"

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from json files at: %s", data_file)
            data = json.load(data_file)
            labels = data[1]
            data = data[0]
            for x, l in zip(data, labels):
                yield self.text_to_instance(x, l)

    @overrides
    def text_to_instance(self, data: dict, relation_type: int = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        N_relations = []
        all_tokens_sentences = []
        for i, K_examples in enumerate(data[self.TRAIN_DATA]):
            toknized_sentences = []
            clean_text_for_debug = []
            for relation in K_examples:
                head_tail = self.create_head_tail_sentence(relation)
                tokenized_tokens = self._tokenizer.tokenize(head_tail)

                field_of_tokens = TextField(tokenized_tokens, self._token_indexers)
                clean_text_for_debug.append(MetadataField(tokenized_tokens))

                toknized_sentences.append(field_of_tokens)
            assert len(toknized_sentences) == len(clean_text_for_debug)

            clean_text_for_debug = ListField(clean_text_for_debug)
            toknized_sentences = ListField(toknized_sentences)

            all_tokens_sentences.append(clean_text_for_debug)
            N_relations.append(toknized_sentences)

        assert len(N_relations) == len(all_tokens_sentences)
        N_relations = ListField(N_relations)
        all_tokens_sentences = ListField(all_tokens_sentences)
        fields = {'sentences': N_relations, "clean_tokens": all_tokens_sentences}

        test_dict = data[self.TEST_DATA]
        head_tail = self.create_head_tail_sentence(test_dict)
        tokenized_tokens = self._tokenizer.tokenize(head_tail)
        test_clean_text_for_debug = MetadataField(tokenized_tokens)
        field_of_tokens = TextField(tokenized_tokens, self._token_indexers)

        fields['test'] = field_of_tokens
        fields['test_clean_text'] = test_clean_text_for_debug

        if relation_type is not None:
            fields['label'] = IndexField(relation_type, N_relations)
        return Instance(fields)

    def create_head_tail_sentence(self, relation: dict) -> str:
        return head_start_token + " " + relation['h'][0] + " " + head_end_token + " " + tail_start_token + ' ' + \
               relation['t'][0] + " " + tail_end_token
