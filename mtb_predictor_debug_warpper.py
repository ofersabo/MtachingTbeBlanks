import json
import shutil
import sys

from allennlp.commands import main

# config_file = "experiments/mtb_config.jsonnet"

# Use overrides to train on CPU.
# overrides = json.dumps({"trainer":{"cuda_device": -1},"iterator": {"type": "basic", "batch_size": 4}})
#
# overrides = json.dumps({"train_data_path": "data/train_small.json","trainer":{"cuda_device": -1},
#   "validation_data_path": "data/val_small.json","iterator": {"type": "basic", "batch_size": 4}})
#
# overrides = json.dumps({"train_data_path": "data/train_small.json",
#   "validation_data_path": "data/val_small.json"})
model_location = "fixed_code/grad_true_two_layers_A/"
output_file = "evaluation.txt"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!

'''
allennlp evaluate [-h] [--output-file OUTPUT_FILE]
                         [--weights-file WEIGHTS_FILE]
                         [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                         [--batch-weight-key BATCH_WEIGHT_KEY]
                         [--extend-vocab]
                         [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                         [--include-package INCLUDE_PACKAGE]
                         archive_file input_file
'''

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    model_location,
    "data/train_10K.json",
    "--output-file", model_location + output_file,
    "--include-package", "my_library"
    # "-o", overrides,
]

main()