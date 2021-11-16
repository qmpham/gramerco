#!/bin/bash

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install -c conda-forge spacy
conda install -c conda-forge cupy
# conda install -c conda-forge numpy
conda install -c conda-forge matplotlib

pip install git+https://github.com/huggingface/transformers
