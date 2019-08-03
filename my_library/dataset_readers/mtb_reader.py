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

head_start_token = 'E1start' # fixme check if this indeed the token they used in the paper
head_end_token = 'E1end' # fixme check if this indeed the token they used in the paper
tail_start_token = 'E2start'
tail_end_token = 'E2end'

TRAIN_DATA = "meta_train"
TEST_DATA = "meta_test"

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def replace_positions(locations):
    locations = [locations[3], locations[4], locations[1], locations[2]]




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
        all_tokens_sentences = []
        for i,K_examples in enumerate(data[TRAIN_DATA]):
            toknized_sentences = []
            sentences_location = []
            tokens_sentences = []
            for rel in K_examples:
                tokenized_tokens = self._tokenizer.tokenize(" ".join(rel["tokens"]))
                head_location,tail_location = self.addStartEntityTokens(tokenized_tokens,rel['h'],rel['t']) #fixme after tokanization tail and head postion change their location
                assert tokenized_tokens[head_location].text == head_start_token
                assert tokenized_tokens[tail_location].text == tail_start_token
                tokens_field = TextField(tokenized_tokens, self._token_indexers)
                locations = MetadataField({"head":head_location,"tail":tail_location})
                tokens_sentences.append(MetadataField(tokenized_tokens))

                sentences_location.append(locations)
                toknized_sentences.append(tokens_field)
            assert len(sentences_location) == len(toknized_sentences) == len(tokens_sentences)

            sentences_location = ListField(sentences_location)
            tokens_sentences = ListField(tokens_sentences)
            toknized_sentences = ListField(toknized_sentences)

            all_tokens_sentences.append(tokens_sentences)
            location_list.append(sentences_location)
            N_relations.append(toknized_sentences)

        assert len(N_relations) == len(location_list) == len(all_tokens_sentences)
        N_relations = ListField(N_relations)
        location_list = ListField(location_list)
        all_tokens_sentences = ListField(all_tokens_sentences)
        fields = {'sentences': N_relations,"locations":location_list,"clean_tokens":all_tokens_sentences}

        test_dict = data[TEST_DATA]
        head_location,tail_location = self.addStartEntityTokens(test_dict)
        locations = MetadataField({"head": head_location, "tail": tail_location})
        tokenized_tokens = self._tokenizer.tokenize(" ".join(test_dict["tokens"]))
        tokens_field = TextField(tokenized_tokens, self._token_indexers)
        fields['test'] = tokens_field
        fields['test_location'] = locations

        if relation_type is not None:
            fields['label'] = IndexField(relation_type,N_relations)
        return Instance(fields)

    def addStartEntityTokens(self,tokens_list, head, tail):
        h_indcies = [x for x in head[2]]  # todo try different approchs on which entity location to choose
        t_indcies = [x for x in tail[2]]
        min_distance = 99999

        for i, x in enumerate(h_indcies):
            for j, y in enumerate(t_indcies):
                if abs(x[0] - y[0]) < min_distance:
                    min_distance = abs(x[0] - y[0])
                    h_start_location, head_end_location, tail_start_location, tail_end_location = (
                    x[0], x[-1] + 1, y[0], y[-1] + 1) if x[0] < y[0] else (y[0], y[-1] + 1, x[0], x[-1] + 1)
        assert h_start_location < tail_start_location
        assert h_start_location < head_end_location
        assert tail_start_location < tail_end_location

        # find correct position after tokenizaion added tokens
        for h_start_location, token in enumerate(tokens_list[h_start_location], h_start_location):
            if token.lower() == head[0].split()[0]:
                break

        for tail_start_location, token in enumerate(tokens_list[tail_start_location], tail_start_location):
            if token.lower() == tail[0].split()[0].lower():
                break

        x = self._tokenizer.tokenize(head_start_token)
        y = self._tokenizer.tokenize(head_end_token)
        z = self._tokenizer.tokenize(tail_start_token)
        w = self._tokenizer.tokenize(tail_end_token)

        tokens_list.insert(h_start_location, x[0])  # arbetrary pick a token for that
        tokens_list.insert(head_end_location + 1, y[0])  # arbetrary pick a token for that
        tokens_list.insert(tail_start_location + 2, z[0])  # arbetrary pick a token for that
        tokens_list.insert(tail_end_location + 3, w[0])  # arbetrary pick a token for that

        return h_start_location, tail_start_location + 2
