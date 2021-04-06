#!/bin/bash -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate tf115
python -c "import tensorflow as tf; print(tf.__version__)"

python $HOME/tinyspeech_harvard/tensorflow/tensorflow/examples/speech_commands/freeze.py \
--wanted_words=hello,thanks \
--window_stride_ms=20 \
--preprocess=micro \
--model_architecture=tiny_conv \
--start_checkpoint=train/tiny_conv.ckpt-1000 \
--save_format=saved_model \
--output_file=hello_thanks
