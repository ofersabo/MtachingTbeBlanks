import json
import shutil
import sys

from allennlp.commands import main

config_file = "experiments/mtb_config.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({"train_data_path": "data/train_small.json","trainer":{"cuda_device": -1},
  "validation_data_path": "data/val_small.json","iterator": {"type": "basic", "batch_size": 2}})
#
# overrides = json.dumps({"train_data_path": "data/train_small.json",
#   "validation_data_path": "data/val_small.json"})

serialization_dir = "/tmp/debug_mtb"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "my_library",
    "-o", overrides,
]

main()