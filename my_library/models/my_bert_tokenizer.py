# from allennlp.data.tokenizers.word_splitter import BertTokenizer
#
#
# import logging
#
#
# logger = logging.getLogger(__name__)
#
# # from allennlp.data.tokenizers.token import show_token
# # from allennlp.data.token_indexers import PretrainedBertIndexer
# from my_library.dataset_readers.mtb_reader import head_start_token
# from my_library.dataset_readers.mtb_reader import tail_start_token

# @WordSplitter.register("ofer-bert-basic")
# class MyBertWordSplitter(BertTokenizer):
#     """
#     The ``BasicWordSplitter`` from the BERT implementation.
#     This is used to split a sentence into words.
#     Then the ``BertTokenIndexer`` converts each word into wordpieces.
#     """
#     def __init__(self, do_lower_case: bool = False,never_split: tuple = None) -> None:
#         # fixme never split should be given as an argument
#         super.__init__()
#
#     @overrides
#     def split_words(self, sentence: str) -> List[Token]:
#         # for text in self.basic_tokenizer.tokenize(sentence):
#             # if text in self.never_split[:5] or text.startswith("a"):
#                 # print(show_token(Token(text)))
#         return [Token(text) for text in self.basic_tokenizer.tokenize(sentence)]
#
