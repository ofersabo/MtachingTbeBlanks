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


@DatasetReader.register("mtb_reader")
class MTBDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

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
        location_list = []
        all_tokens_sentences = []
        for i, K_examples in enumerate(data[self.TRAIN_DATA]):
            toknized_sentences = []
            sentences_location = []
            clean_text_for_debug = []
            for relation in K_examples:
                tokenized_tokens = self._tokenizer.tokenize(" ".join(relation["tokens"]))
                head_location, tail_location = self.addStartEntityTokens(tokenized_tokens, relation['h'], relation['t'])

                assert tokenized_tokens[head_location].text == head_start_token
                assert tokenized_tokens[tail_location].text == tail_start_token

                field_of_tokens = TextField(tokenized_tokens, self._token_indexers)
                locations_of_entities = MetadataField({"head": head_location, "tail": tail_location})
                clean_text_for_debug.append(MetadataField(tokenized_tokens))

                sentences_location.append(locations_of_entities)
                toknized_sentences.append(field_of_tokens)
            assert len(sentences_location) == len(toknized_sentences) == len(clean_text_for_debug)

            sentences_location = ListField(sentences_location)
            clean_text_for_debug = ListField(clean_text_for_debug)
            toknized_sentences = ListField(toknized_sentences)

            all_tokens_sentences.append(clean_text_for_debug)
            location_list.append(sentences_location)
            N_relations.append(toknized_sentences)

        assert len(N_relations) == len(location_list) == len(all_tokens_sentences)
        N_relations = ListField(N_relations)
        location_list = ListField(location_list)
        all_tokens_sentences = ListField(all_tokens_sentences)
        fields = {'sentences': N_relations, "locations": location_list, "clean_tokens": all_tokens_sentences}

        test_dict = data[self.TEST_DATA]
        tokenized_tokens = self._tokenizer.tokenize(" ".join(test_dict["tokens"]))
        head_location, tail_location = self.addStartEntityTokens(tokenized_tokens, test_dict['h'], test_dict['t'])
        test_clean_text_for_debug = MetadataField(tokenized_tokens)
        locations_of_entities = MetadataField({"head": head_location, "tail": tail_location})
        field_of_tokens = TextField(tokenized_tokens, self._token_indexers)

        fields['test'] = field_of_tokens
        fields['test_location'] = locations_of_entities
        fields['test_clean_text'] = test_clean_text_for_debug

        if relation_type is not None:
            fields['label'] = IndexField(relation_type, N_relations)
        return Instance(fields)

    def addStartEntityTokens(self, tokens_list, head_full_data, tail_full_data):
        if len(head_full_data[0]) > len(tail_full_data[0]): # this is for handling nested tail and head entities
            #for example: head = NEC and tail = NEC corp
            # solution, make sure no overlapping entities mention
            head_start_location, head_end_location = self.find_locations(head_full_data, tokens_list)
            tail_start_location, tail_end_location = self.find_locations(tail_full_data, tokens_list)
            if tail_start_location[0]>= head_start_location[0] and tail_start_location[0] <= head_end_location[0]:
                tail_end_location, tail_start_location = self.deny_overlapping(tokens_list, head_end_location,
                                                                               tail_full_data)

        else:
            tail_start_location, tail_end_location = self.find_locations(tail_full_data, tokens_list)
            head_start_location, head_end_location = self.find_locations(head_full_data, tokens_list)
            if head_start_location[0] >= tail_start_location[0] and head_start_location[0] <= tail_end_location[0]:
                head_end_location, head_start_location = self.deny_overlapping(tokens_list, tail_end_location,head_full_data)

        # todo try different approchs on which entity location to choose
        h_start_location, head_end_location, tail_start_location, tail_end_location = find_closest_distance_between_entities \
            (head_start_location, head_end_location, tail_start_location, tail_end_location)

        x = self._tokenizer.tokenize(head_start_token)
        y = self._tokenizer.tokenize(head_end_token)
        z = self._tokenizer.tokenize(tail_start_token)
        w = self._tokenizer.tokenize(tail_end_token)

        offset_tail = 2 * (tail_start_location > h_start_location)
        tokens_list.insert(h_start_location, x[0])  # arbetrary pick a token for that
        tokens_list.insert(head_end_location + 1 + 1, y[0])  # arbetrary pick a token for that
        tokens_list.insert(tail_start_location + offset_tail, z[0])  # arbetrary pick a token for that
        tokens_list.insert(tail_end_location + 2 + offset_tail, w[0])  # arbetrary pick a token for that

        return h_start_location + 2 - offset_tail, tail_start_location + offset_tail

    def deny_overlapping(self, tokens_list, longest_entity_end_location, shortest_entity_full_data):
        start_location, end_location = self.find_locations(shortest_entity_full_data, tokens_list[longest_entity_end_location[0]+1:])
        start_location[0] = start_location[0] + longest_entity_end_location[0]
        end_location[0] = end_location[0] + longest_entity_end_location[0]
        return end_location, start_location

    def return_lower_text_from_tokens(self, tokens):
        return list(map(lambda x: x.text.lower(), tokens))

    def compare_two_token_lists(self, x, y):
        return self.return_lower_text_from_tokens(x) == self.return_lower_text_from_tokens(y)

    def spacy_work_toknizer(self,text):
        return list(map(lambda x: x.text,self.spacy_splitter.split_words(text)))

    def find_locations(self, head_full_data, token_list):
        end_location, start_location = self._find_entity_name(token_list, head_full_data)
        if len(end_location) == 0 or len(start_location) == 0:
            end_location, start_location = self._find_entity_name(token_list, head_full_data, True)

        assert len(start_location) == len(end_location)
        assert len(start_location) == len(head_full_data[2])

        return start_location, end_location

    def _find_entity_name(self, token_list, head_full_data,use_spacy_toknizer_before = False):
        if use_spacy_toknizer_before:
            spacy_head_tokens = self.spacy_work_toknizer(head_full_data[0])
            head = self._tokenizer.tokenize(" ".join(spacy_head_tokens))
        else:
            head = self._tokenizer.tokenize(" ".join([head_full_data[0]]))
        start_head_entity_name = head[0]
        start_location = []
        end_location = []
        for i, token in enumerate(token_list):
            if self.compare_two_token_lists([token], [start_head_entity_name]):
                if self.compare_two_token_lists(token_list[i:i + len(head)], head):
                    start_location.append(i)
                    end_location.append(i + len(head) - 1)
                    if len(start_location) == len(head_full_data[2]):
                        break
        return end_location, start_location
