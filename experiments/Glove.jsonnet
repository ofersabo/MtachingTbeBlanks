local cuda = [0,1,2,3];
//local cuda = [0,1,2,3,4,5,6,7];
local batch_size = 20;
local full_training = true;
local small_dataset = false;
local lr_with_find = 0.00001;

// Configuration for the basic QANet model from "QANet: Combining Local
// Convolution with Global Self-Attention for Reading Comprehension"
// (https://arxiv.org/abs/1804.09541).
{
    "dataset_reader": {
        "type": "only_entities_reader",
        "tokenizer": {"word_splitter": "just_spaces"
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },
    },
    "vocabulary": {
        "pretrained_files": {
            // This embedding file is created from the Glove 840B 300d embedding file.
            // We kept all the original lowercased words and their embeddings. But there are also many words
            // with only the uppercased version. To include as many words as possible, we lowered those words
            // and used the embeddings of uppercased words as an alternative.
            "tokens": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.lower.converted.zip"
        },
        "only_include_pretrained_words": true
    },
    "train_data_path":  if small_dataset then "data/train_small.json" else if full_training then "data/train_100K.json" else "data/train_10K.json",
    "validation_data_path": if small_dataset then "data/val_small.json" else if full_training then "data/val_5K.json" else "data/val_5K.json",
     "model": {
        "type": "Embedding_and_bilstm",
        "number_of_layers": 2,
        "skip_connection": true,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.lower.converted.zip",
                    "embedding_dim": 300,
                    "trainable": true
                },
            }
        },
        "regularizer": [["liner_layer", {"type": "l2", "alpha": 1e-03}], [".*", {"type": "l2", "alpha": 1e-07}]]
//        "num_highway_layers": 2,
//        "phrase_layer": {
//            "type": "qanet_encoder",
//            "input_dim": 128,
//            "hidden_dim": 128,
//            "attention_projection_dim": 128,
//            "feedforward_hidden_dim": 128,
//            "num_blocks": 1,
//            "num_convs_per_block": 4,
//            "conv_kernel_size": 7,
//            "num_attention_heads": 8,
//            "dropout_prob": 0.1,
//            "layer_dropout_undecayed_prob": 0.1,
//            "attention_dropout_prob": 0
//        },
//        "matrix_attention_layer": {
//            "type": "linear",
//            "tensor_1_dim": 128,
//            "tensor_2_dim": 128,
//            "combination": "x,y,x*y"
//        },
//        "modeling_layer": {
//            "type": "qanet_encoder",
//            "input_dim": 128,
//            "hidden_dim": 128,
//            "attention_projection_dim": 128,
//            "feedforward_hidden_dim": 128,
//            "num_blocks": 7,
//            "num_convs_per_block": 2,
//            "conv_kernel_size": 5,
//            "num_attention_heads": 8,
//            "dropout_prob": 0.1,
//            "layer_dropout_undecayed_prob": 0.1,
//            "attention_dropout_prob": 0
//        },
//        "dropout_prob": 0.1,
//        "regularizer": [
//            [
//                ".*",
//                {
//                    "type": "l2",
//                    "alpha": 1e-07
//                }
//            ]
//        ]
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
    "num_epochs": 100,
//    "grad_norm": 1.0,
    "patience": 7,
    "cuda_device": cuda
  }
}
