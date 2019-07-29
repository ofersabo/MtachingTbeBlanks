from typing import Dict
from typing import List
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import ListField, IndexField, MetadataField,Field

head_start_token = '[E1start]' # fixme check if this indeed the token they used in the paper
head_end_token = '[E1end]' # fixme check if this indeed the token they used in the paper
tail_start_token = '[E2start]'
tail_end_token = '[E2end]'

TRAIN_DATA = "meta_train"
TEST_DATA = "meta_test"

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def replace_positions(locations):
    locations = [locations[3], locations[4], locations[1], locations[2]]


def addStartEntityTokens(d):
    head = d['h']
    tail = d['t']
    tokens_list = d['tokens']
    h_indcies = [x for x in head[2]]  # todo try different approchs on which entity location to choose
    t_indcies = [x for x in tail[2]]
    min_distance = 99999


    for i,x in enumerate(h_indcies):
        for j,y in enumerate(t_indcies):
            if abs(x[0] - y[0]) < min_distance:
                min_distance = abs(x[0] - y[0])
                h_start_location,head_end_location,tail_start_location,tail_end_location = (x[0],x[-1] + 1,y[0],y[-1] + 1) if x[0] < y[0] else (y[0],y[-1] + 1,x[0],x[-1] +1)
    assert  h_start_location < tail_start_location
    assert h_start_location <head_end_location
    assert tail_start_location < tail_end_location
    tokens_list.insert(h_start_location, head_start_token)  # arbetrary pick a token for that
    tokens_list.insert(head_end_location +1, head_end_token )  # arbetrary pick a token for that
    tokens_list.insert(tail_start_location+2, tail_start_token)  # arbetrary pick a token for that
    tokens_list.insert(tail_end_location+3, tail_end_token)  # arbetrary pick a token for that

    return h_start_location, tail_start_location+2

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

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path),"r") as data_file:
            logger.info("Reading instances from json files at: %s", data_file)
            data = json.load(data_file)
            labels = data[1]
            data = data[0]
            for x,l in zip(data,labels):
                    yield self.text_to_instance(x, l)

    @overrides
    def text_to_instance(self, data: dict, relation_type: int = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        N_relations = []
        location_list = []
        for K_examples in data[TRAIN_DATA]:
            sentences_on_relations = []
            K_sentences_location = []
            for rel in K_examples:
                head_location,tail_location = addStartEntityTokens(rel)
                tokenized_tokens = self._tokenizer.tokenize(" ".join(rel["tokens"]))
                tokens_field = TextField(tokenized_tokens, self._token_indexers)
                locations = MetadataField({"head":head_location,"tail":tail_location})
                # tail_location = IndexField(tail_location,tokens_field)
                K_sentences_location.append(locations)
                sentences_on_relations.append(tokens_field)
            assert len(K_sentences_location) == len(sentences_on_relations)
            K_sentences_location = ListField(K_sentences_location)
            sentences_on_relations = ListField(sentences_on_relations)

            location_list.append(K_sentences_location)
            N_relations.append(sentences_on_relations)

        assert len(N_relations) == len(location_list)
        N_relations = ListField(N_relations)
        location_list = ListField(location_list)
        fields = {'sentences': N_relations,"locations":location_list}

        test_dict = data[TEST_DATA]
        head_location,tail_location = addStartEntityTokens(test_dict)
        locations = MetadataField({"head": head_location, "tail": tail_location})
        tokenized_tokens = self._tokenizer.tokenize(" ".join(test_dict["tokens"]))
        tokens_field = TextField(tokenized_tokens, self._token_indexers)
        fields['test'] = tokens_field
        fields['test_location'] = locations

        if relation_type is not None:
            fields['label'] = IndexField(relation_type,N_relations)
        return Instance(fields)

