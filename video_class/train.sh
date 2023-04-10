PYTHON=./mario/bin/python

nohup $PYTHON -u train2.py --batchsize 256 --CUDA 0 > 20220207_train.log 2>&1 &

