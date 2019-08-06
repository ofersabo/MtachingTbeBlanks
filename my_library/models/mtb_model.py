from typing import Dict, List
import collections
import logging
import math
import allennlp
import torch
from overrides import overrides
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from pytorch_pretrained_bert import BertForQuestionAnswering as HuggingFaceBertQA
from pytorch_pretrained_bert import BertModel as BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from torch.autograd import Variable
from allennlp.common import JsonDict
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.training.metrics import CategoricalAccuracy
from my_library.dataset_readers.mtb_reader import head_start_token, tail_start_token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


BERT_LARGE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                     "hidden_act": "gelu",
                     "hidden_dropout_prob": 0.1,
                     "hidden_size": 1024,
                     "initializer_range": 0.02,
                     "intermediate_size": 4096,
                     "max_position_embeddings": 512,
                     "num_attention_heads": 16,
                     "num_hidden_layers": 24,
                     "type_vocab_size": 2,
                     "vocab_size": 30522
                    }

BERT_BASE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0.1,
                    "hidden_size": 768,
                    "initializer_range": 0.02,
                    "intermediate_size": 3072,
                    "max_position_embeddings": 512,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "type_vocab_size": 2,
                    "vocab_size": 30522
                   }


@Model.register('bert_for_mtb')
class BertEmbeddingsMTB(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 metrics: Dict[str, allennlp.training.metrics.Metric] = None,
                 # cuda_device : int = 1,
                 bert_model: str = None,
                 # ,load_model_from: str = None
                 ) -> None:
        super().__init__(vocab)
        self.embbedings = text_field_embedder
        self.bert_type_model = BERT_BASE_CONFIG if "base" in bert_model else BERT_LARGE_CONFIG
        self.extractor = EndpointSpanExtractor(input_dim=self.bert_type_model['hidden_size'], combination="x,y")
        self.crossEntropyLoss   = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics or {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size']*2,1000)
        self.relation_layer_norm = torch.nn.LayerNorm(torch.Size([self.bert_type_model['hidden_size']*2]), elementwise_affine=True)
        self.head_token_index = 1
        self.tail_token_index = 3
        # if load_model_from:
        #     self.load(self.Model.get_params() ,serialization_dir=load_model_from,cuda_device=cuda_device)


    @overrides
    def forward(self,  sentences, locations, test, test_location,clean_tokens,test_clean_text,
                label = None) -> Dict[str, torch.Tensor]:

        tensor_of_matrices = torch.zeros(0,5,self.bert_type_model['hidden_size']*2).to(self.device)
        test_matrix = torch.zeros(0,self.bert_type_model['hidden_size']*2).to(self.device)

        bert_context_for_relation = self.embbedings(sentences)
        test_bert = self.embbedings(test)

        self.debug_issue(bert_context_for_relation, sentences, test, test_bert)

        for batch_input in range(bert_context_for_relation.size(0)):
            matrix_all_N_relation = torch.zeros(0,self.bert_type_model['hidden_size']*2).to(self.device)
            for i in range(bert_context_for_relation.size(1)):
                # toekns_list = self.reassemble_sentence_for_debug(sentences, batch_input, i)

                head, tail = self.get_head_tail_locations(sentences['bert'][batch_input, i, 0, :])
                concat_represntentions = self.extract_embeddings_of_start_tokens(bert_context_for_relation, i ,
                                                                             batch_input, head, tail)

                concat_represntentions = self.renorm_vector(concat_represntentions)
                matrix_all_N_relation = torch.cat((matrix_all_N_relation, concat_represntentions),0).to(self.device)

            matrix_all_N_relation = matrix_all_N_relation.unsqueeze(0)
            tensor_of_matrices = torch.cat((tensor_of_matrices,matrix_all_N_relation),0).to(self.device)

            # test query
            head, tail = self.get_head_tail_locations(test['bert'][batch_input, :])
            if head > test_bert.size(1) or tail > test_bert.size(1):
                logger.warning("Problem")
                toekns_list = self.reassemble_sentence_for_debug(test, batch_input, i)
                test_bert = self.embbedings(test)
            test_concat = self.extract_embeddings_of_start_tokens(test_bert, i, batch_input, head, tail)
            test_concat = self.renorm_vector(test_concat)

            test_matrix = torch.cat((test_matrix, test_concat), 0).to(self.device)

        test_matrix = test_matrix.unsqueeze(1)
        tensor_of_matrices = tensor_of_matrices.permute(0,2,1)
        scores = torch.matmul(test_matrix,tensor_of_matrices).squeeze(1).to(self.device)
        output_dict = {"scores": scores}
        if label is not None:
            label = label.squeeze(1)
            loss = self.crossEntropyLoss(scores, label)
            output_dict["loss"] = loss
            for metric in self.metrics.values():
                metric(scores, label)

        return output_dict

    def renorm_vector(self, concat_represntentions):
        return torch.renorm(concat_represntentions, 2, 0, 1)
        # return self.relation_layer_norm(concat_represntentions)

    def debug_issue(self, bert_context_for_relation, sentences, test, test_bert):
        if bert_context_for_relation.size(-2) != sentences['bert'].size(-1):
            bert_context_for_relation = self.embbedings(sentences)
            logger.warning("Problem")
            exit()
        if test_bert.size(-2) != test['bert'].size(-1):
            logger.warning("Problem")
            test_bert = self.embbedings(test)
            exit()

    def reassemble_sentence_for_debug(self, sentences, batch_input, i):
        token_to_index = self.vocab._token_to_index['bert']
        index_to_token = self.vocab._index_to_token['bert']
        try:
            this_sentence = sentences['bert'][batch_input, i, 0, :]
        except IndexError:
            this_sentence = sentences['bert'][batch_input]

        tokens = []
        for i in this_sentence:
            tokens.append(index_to_token[i.item()])
        return tokens

    def get_head_tail_locations(self, sentence):
        for i,index_value in enumerate(sentence):
            if index_value.item() == self.tail_token_index:
                tail = i
            if index_value.item() == self.head_token_index:
                head = i
        assert type(head) is int
        assert type(tail) is int
        return head, tail

    def assert_head_tail_correct_location(self, batch_input, clean_tokens, head, i, tail):
        assert clean_tokens[batch_input][i][0][tail].text == tail_start_token  # fixme remove
        assert clean_tokens[batch_input][i][0][head].text == head_start_token

    def extract_embeddings_of_start_tokens(self, relation_representation, i ,batch_input, head, tail):
        indices = Variable(torch.LongTensor([[head, tail]])).to(self.device)
        try:
            x = relation_representation[batch_input, i, :, :, :].to(self.device)
            length_of_seq = x.size(1)
        except IndexError:
            x = relation_representation[batch_input, :, :].to(self.device)
            length_of_seq = x.size(0)

        assert length_of_seq > head
        assert length_of_seq > tail
        concat_represntentions = self.extractor(x, indices).to(self.device)
        return concat_represntentions

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
