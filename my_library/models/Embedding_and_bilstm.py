from typing import Dict, List
import collections
import logging
import math
import allennlp
import torch
from my_library.dataset_readers.only_entities_reader import head_start_token,head_end_token,tail_start_token,tail_end_token
from overrides import overrides
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from pytorch_pretrained_bert import BertForQuestionAnswering as HuggingFaceBertQA
from pytorch_pretrained_bert import BertModel as BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from allennlp.nn.regularizers.regularizers import L2Regularizer
from torch.autograd import Variable
from allennlp.common import JsonDict
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn import InitializerApplicator, RegularizerApplicator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

linear = "linear"

@Model.register('Embedding_and_bilstm')
class EmbeddingsMTB(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 metrics: Dict[str, allennlp.training.metrics.Metric] = None,
                 number_of_layers: int = 2,
                 number_of_linear_layers: int = 2,
                 renorm_method: str = None,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 bert_model: str = None,
                 ) -> None:
        super().__init__(vocab,regularizer)
        self.embbedings = text_field_embedder
        self.hidden_size = 250
        self.bilstm = torch.nn.LSTM(input_size=300,hidden_size=self.hidden_size,num_layers=number_of_layers,batch_first=True,
                                    bidirectional=True, dropout=0.2)
        self.extractor = EndpointSpanExtractor(input_dim=self.hidden_size, combination="x,y")
        self.crossEntropyLoss   = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics or {
            "accuracy": CategoricalAccuracy()
        }
        self.first_liner_layer = torch.nn.Linear(self.hidden_size*2*2,self.hidden_size*2*2) # twiche double, firest is because bidirectional and second because concat
        self.second_liner_layer = torch.nn.Linear(self.hidden_size*2*2,self.hidden_size*2*2)
        self.do_skip_connection = skip_connection

        self.number_of_linear_layers = number_of_linear_layers
        self.relation_layer_norm = torch.nn.LayerNorm(torch.Size([self.hidden_size*2]), elementwise_affine=True)
        self.tanh = torch.nn.Tanh()
        self.drop_layer = torch.nn.Dropout(p=0.2)
        self.renorm_method = renorm_method or linear
        for t in [head_start_token,head_end_token,tail_start_token,tail_end_token]:
            index = self.vocab.add_token_to_namespace(t)
            if t == head_start_token:
                self.head_token_index = index
            elif t == tail_start_token:
                self.tail_token_index = index

    @overrides
    def forward(self,  sentences, test, clean_tokens,test_clean_text,
                label = None) -> Dict[str, torch.Tensor]:

        tensor_of_matrices = torch.zeros(0,5,self.hidden_size*2*2).to(self.device)
        test_matrix = torch.zeros(0,self.hidden_size*2*2).to(self.device)

        glove = self.embbedings(sentences)
        glove = glove.squeeze(2)
        glove_for_rnn = glove.view(glove.size(0) * glove.size(1),glove.size(2),glove.size(3))
        lstm_N_sentences,_ = self.bilstm(glove_for_rnn)
        lstm_N_sentences = lstm_N_sentences.view(glove.size(0),glove.size(1),glove.size(2),-1)
        # lstm_N_sentences = torch.zeros(0,5,glove.size(2),2*self.hidden_size).to(self.device)
        # for g in glove:
        #     g_after_lstm,_ = self.bilstm(g)
        #     lstm_N_sentences = torch.cat((lstm_N_sentences,g_after_lstm),dim=0)

        query_embedding = self.embbedings(test)
        test_bilstm,_ = self.bilstm(query_embedding)
        # self.debug_issue(lstm_N_sentences, sentences, test, test_bert)
        for batch_input in range(lstm_N_sentences.size(0)):
            matrix_all_N_relation = torch.zeros(0,self.hidden_size*2*2).to(self.device)
            for i in range(lstm_N_sentences.size(1)):

                head, tail = self.get_head_tail_locations(sentences['tokens'][batch_input, i, 0, :])
                concat_represntentions = self.extract_embeddings_of_start_tokens(lstm_N_sentences, i ,
                                                                             batch_input, head, tail)

                final_represnetation = self.renorm_vector(concat_represntentions)
                matrix_all_N_relation = torch.cat((matrix_all_N_relation, final_represnetation),0).to(self.device)

            matrix_all_N_relation = matrix_all_N_relation.unsqueeze(0)
            tensor_of_matrices = torch.cat((tensor_of_matrices,matrix_all_N_relation),0).to(self.device)

            # test query
            head, tail = self.get_head_tail_locations(test['tokens'][batch_input, :])
            # test_bert = self.debug_query_sentence(test, test_bert, head, tail, batch_input, i)

            test_concat = self.extract_embeddings_of_start_tokens(test_bilstm, i, batch_input, head, tail)
            final_query_representation = self.renorm_vector(test_concat)

            test_matrix = torch.cat((test_matrix, final_query_representation), 0).to(self.device)

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

    def debug_query_sentence(self, test, test_bert, head, tail, batch_input, i):
        if head > test_bert.size(1) or tail > test_bert.size(1):
            logger.warning("Problem")
            toekns_list = self.reassemble_sentence_for_debug(test, batch_input, i)
            test_bert = self.embbedings(test)
        return test_bert

    def renorm_vector(self, concat_represntentions):
        if self.renorm_method != linear:
            return torch.renorm(concat_represntentions, 2, 0, 1)

        # return self.relation_layer_norm(concat_represntentions)
        x = self.first_liner_layer(concat_represntentions)
        x = self.tanh(x)
        x = self.drop_layer(x)
        x = self.second_liner_layer(x)
        if self.do_skip_connection:
            x = x + concat_represntentions

        return x

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
        head = None
        tail = None
        for i,index_value in enumerate(sentence):
            if index_value.item() == self.tail_token_index:
                assert tail is None
                tail = i
                if head is not None:
                    return head,tail
            if index_value.item() == self.head_token_index:
                assert head is None
                head = i
                if tail is not None:
                    return head, tail

        return head, tail

    def assert_head_tail_correct_location(self, batch_input, clean_tokens, head, i, tail):
        assert clean_tokens[batch_input][i][0][tail].text == tail_start_token  # fixme remove
        assert clean_tokens[batch_input][i][0][head].text == head_start_token

    def extract_embeddings_of_start_tokens(self, relation_representation, i ,batch_input, head, tail):
        indices = Variable(torch.LongTensor([[head, tail]])).to(self.device)
        try:
            x = relation_representation[batch_input, i, :, :].to(self.device)
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
