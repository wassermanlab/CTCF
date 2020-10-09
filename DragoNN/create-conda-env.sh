#!/usr/bin/env bash

# i.e. enable conda (de)activate
eval "$(conda shell.bash hook)"

# Create DragoNN environment
conda create -n dragonn -c bioconda biopython deeptools pybedtools
conda activate dragonn
conda install -c conda-forge bazel=0.26.1

# Install DragoNN
conda activate dragonn
pip install dragonn

# Build tensorflow from source:
# https://www.tensorflow.org/install/source#ubuntu
conda activate dragonn
pip uninstall tensorflow-gpu
wget https://github.com/tensorflow/tensorflow/archive/v1.15.4.tar.gz
tar xvfz v1.15.4.tar.gz
cd tensorflow-1.15.4/
./configure
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-2.3.1-cp37-cp37m-linux_x86_64.whl 

# Errors:
# 2020-10-05 08:47:24.764380: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.11.0
# Using TensorFlow backend.
# loading sequence data...
# initializing model...
# Traceback (most recent call last):
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/bin/dragonn", line 8, in <module>
#     sys.exit(main())
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/dragonn/__init__.py", line 224, in main
#     command_functions[command](**args)
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/dragonn/__init__.py", line 113, in main_train
#     model = SequenceDNN(seq_length=X_train.shape[-1], **kwargs)
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/dragonn/models.py", line 120, in __init__
#     W_regularizer=l1(L1), b_regularizer=l1(L1)))
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/keras/engine/sequential.py", line 166, in add
#     layer(x)
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py", line 75, in symbolic_fn_wrapper
#     return func(*args, **kwargs)
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/keras/engine/base_layer.py", line 446, in __call__
#     self.assert_input_compatibility(inputs)
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/keras/engine/base_layer.py", line 310, in assert_input_compatibility
#     K.is_keras_tensor(x)
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py", line 695, in is_keras_tensor
#     if not is_tensor(x):
#   File "/mnt/md1/home/oriol/.conda/envs/dragonn/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py", line 703, in is_tensor
#     return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)
# AttributeError: module 'tensorflow.python.framework.ops' has no attribute '_TensorLike'
