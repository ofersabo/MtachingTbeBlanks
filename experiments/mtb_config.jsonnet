//local cuda = [0,1,2,3];
local cuda = [0,1,2,3,4,5,6,7];
//local bert_type = 'bert-base-cased';
 local bert_type = 'bert-large-cased';
local batch_size = 10;
local full_training = true;
local small_dataset = false;
local lr_with_find = 0.00001;

{
  "dataset_reader": {
    "type": "mtb_reader",
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "ofer-bert-basic",
        "do_lower_case": false
      }
    },
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": bert_type,
          "do_lowercase": false,
          "use_starting_offsets": false
      }
    }
  },
  "train_data_path":  if small_dataset then "data/train_small.json" else if full_training then "data/train_1M.json" else "data/train_10K.json",
  "validation_data_path": if small_dataset then "data/val_small.json" else if full_training then "data/val_5K.json" else "data/val_5K.json",
  "model": {
    "type": "bert_for_mtb",
    "bert_model": bert_type,
    "number_of_linear_layers": 2,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {

            "bert": ["bert"]
        },
        "token_embedders": {
            "bert": {
              "type": "bert-pretrained",
              "pretrained_model":  bert_type,
              "top_layer_only": true,
              "requires_grad": true
            }
        }
    },
    "regularizer": [["liner_layer", {"type": "l2", "alpha": 1e-03}], [".*", {"type": "l2", "alpha": 1e-07}]]

  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size
  },
  "trainer": {
//  "regularizers": "l2",
    "optimizer": {
        "type": "adam",
        "lr": lr_with_find
    },
    "num_serialized_models_to_keep": 3,
    "validation_metric": "+accuracy",
    "num_epochs": 25,
//    "grad_norm": 1.0,
    "patience": 7,
    "cuda_device": cuda
  }
}