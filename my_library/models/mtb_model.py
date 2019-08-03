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
        # self.use_cuda = torch.cuda.is_available()
        # self.cuda_device = "cuda:" +str(cuda_device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics or {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size']*2,1000)
        self.relation_layer_norm = torch.nn.LayerNorm(torch.Size([self.bert_type_model['hidden_size']*2]), elementwise_affine=False)

        # if load_model_from:
        #     self.load(self.Model.get_params() ,serialization_dir=load_model_from,cuda_device=cuda_device)


    @overrides
    def forward(self,  sentences, locations, test, test_location,clean_tokens,
                label = None) -> Dict[str, torch.Tensor]:

        tensor_of_matrices = torch.zeros(0,5,self.bert_type_model['hidden_size']*2).to(self.device)
        test_matrix = torch.zeros(0,self.bert_type_model['hidden_size']*2).to(self.device)
        relation_representation = self.embbedings(sentences)
        test_proxy = self.embbedings(test)

        for batch_input in range(relation_representation.size(0)):
            matrix_all_N_relation = torch.zeros(0,self.bert_type_model['hidden_size']*2).to(self.device)
            for i in range(relation_representation.size(1)):
                head, tail = locations[batch_input][i][0]['head'], locations[batch_input][i][0]['tail']
                tail = tail + 1
                indices = Variable(torch.LongTensor([[head, tail]])).to(self.device)
                x = relation_representation[batch_input, i, :, :, :].to(self.device)
                concat_represntentions = self.extractor(x,indices).to(self.device)
                concat_represntentions = torch.renorm(concat_represntentions, 2, 0, 1)
                matrix_all_N_relation = torch.cat((matrix_all_N_relation, concat_represntentions),0).to(self.device)

            tensor_of_matrices = torch.cat((tensor_of_matrices,matrix_all_N_relation.unsqueeze(0)),0).to(self.device)

            head, tail = test_location[batch_input]['head'], test_location[batch_input]['tail']
            indices = Variable(torch.LongTensor([[head, tail]])).to(self.device)
            x = test_proxy[batch_input, :, :].to(self.device)
            test_concat = self.extractor(x, indices)
            test_concat = torch.renorm(test_concat, 2, 0, 1)
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

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
