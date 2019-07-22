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
from allennlp.data.fields import ListField, IndexField

head_start = '[CLS]' # fixme check if this indeed the token they used in the paper
tail_start = '[CLS]'

TRAIN_DATA = "meta_train"
TEST_DATA = "meta_test"

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def addStartEntityTokens(d):
    head = d['h']
    tail = d['t']
    tokens_list = d['tokens']
    h_index = [x[0] for x in head[2]]  # todo try different approchs on which entity location to choose
    t_index = [x[0] for x in tail[2]]
    min_distance = 99999


    for x in h_index:
        for y in t_index:
            if abs(x - y) < min_distance:
                min_distance = abs(x - y)
                h_location = x
                tail_location = y

    tokens_list.insert(h_location, head_start)  # arbetrary pick a token for that
    tokens_list.insert(tail_location + (h_location < tail_location) * 1, tail_start)  # add 1 because of adding token
    # logger.info(tokenized_tokens)


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
        for K_examples in data[TRAIN_DATA]:
            sentences_on_relations = []
            for rel in K_examples:
                addStartEntityTokens(rel)
                tokenized_tokens = self._tokenizer.tokenize(" ".join(rel["tokens"]))
                tokens_field = TextField(tokenized_tokens, self._token_indexers)
                sentences_on_relations.append(tokens_field)
            sentences_on_relations = ListField(sentences_on_relations)
            N_relations.append(sentences_on_relations)
        N_relations = ListField(N_relations)
        fields = {'sentences': N_relations}

        test_dict = data[TEST_DATA]
        addStartEntityTokens(test_dict)
        tokenized_tokens = self._tokenizer.tokenize(" ".join(test_dict["tokens"]))
        tokens_field = TextField(tokenized_tokens, self._token_indexers)
        fields['test'] = tokens_field

        if relation_type is not None:
            fields['label'] = LabelField(str(relation_type))
        return Instance(fields)

