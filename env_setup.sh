#!/bin/bash
#Chaning the paths to cache in bitbucket
export HF_HOME="/vol/bitbucket/${USER}/.cache"
export HF_DATASETS_CACHE="/vol/bitbucket/${USER}/.cache/huggingface/datasets"
export DIFFUSERS_CACHE="/vol/bitbucket/${USER}/.cache/huggingface/diffusers"