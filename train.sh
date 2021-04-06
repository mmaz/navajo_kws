#!/bin/bash -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate tf115
python -c "import tensorflow as tf; print(tf.__version__)"

# run from tinyspeech_harvard/navajo parent directory
python ../tensorflow/tensorflow/examples/speech_commands/train.py \
	--data_dir=wav_data/ \
	--wanted_words=hello,thanks \
	--silence_percentage=25 --unknown_percentage=25 \
	--preprocess=micro \
	--window_stride=20 \
	--model_architecture=tiny_conv \
	--how_many_training_steps=12000,3000 \
	--learning_rate=0.001,0.0001 \
	--train_dir=train/ --summaries_dir=logs/ --verbosity=INFO --eval_step_interval=1000 --save_step_interval=1000
