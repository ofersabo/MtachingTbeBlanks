from allennlp.data.tokenizers.word_splitter import *
from allennlp.data.tokenizers.token import show_token
# from allennlp.data.token_indexers import PretrainedBertIndexer

@WordSplitter.register("ofer-bert-basic")
class MyBertWordSplitter(WordSplitter):
    """
    The ``BasicWordSplitter`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """
    def __init__(self, do_lower_case: bool = True,never_split: tuple = None) -> None:
        if never_split:
            self.never_split = never_split
            self.basic_tokenizer = BertTokenizer(do_lower_case,never_split)
        else:
            never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[unused1]","[unused2]","[unused3]","[unused4]")
            self.never_split = never_split
            self.basic_tokenizer = BertTokenizer(do_lower_case,never_split)

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # for text in self.basic_tokenizer.tokenize(sentence):
            # if text in self.never_split[:5] or text.startswith("a"):
                # print(show_token(Token(text)))
        return [Token(text) for text in self.basic_tokenizer.tokenize(sentence)]