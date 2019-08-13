from overrides import overrides
import numpy as np
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import json

json_file = open("data/fewrel_mapping.json","r")
id2rel_name = json.load(json_file)
def softmax(x):
    x = np.array(x)
    x -= np.max(x)
    deno = sum(np.exp(x))
    proba = np.exp(x) / deno
    np.set_printoptions(precision=2)
    new_datalist = ["{:.2f}".format(value) for value in proba]
    return str(new_datalist)

@Predictor.register('mtb-predictor')
class MTBClassifierPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        mapping_set_index_to_realtion_type,gold_pred = self.extract_mapping_and_correct_answer(inputs)
        output_dict = {}

        instance = self._json_to_instance(inputs)
        scores = self.predict_instance(instance)['scores']
        prediction = np.argmax(scores)
        answer: str = "correct" if prediction == gold_pred else "wrong"
        output_dict["answer"] = answer
        output_dict["prediction"] = str(prediction)
        output_dict["gold"] = str(gold_pred)
        output_dict["probability"] = str(softmax(scores))

        for i,relation in enumerate(inputs[self._dataset_reader.TRAIN_DATA]):
            relation = relation[0]
            output_dict["sentence_"+str(i)] = " ".join(relation["tokens"])
            output_dict["sentence_"+str(i)+"_head"] = relation["h"][0]
            output_dict["sentence_"+str(i)+"_tail"] = relation["t"][0]
            output_dict["sentence_"+str(i)+"_relation"] = id2rel_name[mapping_set_index_to_realtion_type[i].lower()]

        relation = inputs[self._dataset_reader.TEST_DATA]
        output_dict["query"] = " ".join(relation["tokens"])
        output_dict["query_head"] = relation["h"][0]
        output_dict["query_tail"] = relation["t"][0]
        true_relation = id2rel_name[mapping_set_index_to_realtion_type[gold_pred].lower()]
        output_dict["correct_relation"] = true_relation

        return output_dict

    def extract_mapping_and_correct_answer(self, inputs):
        this_set_mapping = []
        for i,examples in enumerate(inputs[self._dataset_reader.TRAIN_DATA]):
            this_set_mapping.append(examples[0])
            inputs[self._dataset_reader.TRAIN_DATA][i] = examples[1]
        x = inputs[self._dataset_reader.TEST_DATA]
        correct = x[0]
        inputs[self._dataset_reader.TEST_DATA] = x[1]
        return this_set_mapping,correct


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        #def text_to_instance(self, data: dict, relation_type: int = None) -> Instance:
        this_instance = self._dataset_reader.text_to_instance(data=json_dict)
        return this_instance
