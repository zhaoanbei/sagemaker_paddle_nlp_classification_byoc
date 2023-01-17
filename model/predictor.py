# -*- coding: utf-8 -*-
import sys
import json
import boto3
import os
import warnings
import numpy as np
import argparse
import flask
import math
from pprint import pprint
import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore",category=FutureWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

# The flask app for serving predictions
app = flask.Flask(__name__)
print ("<<< files under opt/program", os.listdir('/opt/program/'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.", )
    parser.add_argument(
        "--use_fp16",
        action='store_true',
        help="Whether to use fp16 inference, only takes effect when deploying on gpu.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="The maximum input sequence length. Sequences longer than this will be split automatically.",
    )
    parsed, unknown = parser.parse_known_args(args=[]) # this is an 'internal' method

    try:
        for arg in unknown:
            if arg.startswith(("-", "--")):
                # you can pass any arguments to add_argument
                parser.add_argument(arg, type=str)
    except:
        print(unknown)

    args = parser.parse_args(args=[])
    return args

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def model_fn(model_dir):
    params_path = model_dir       # 模型checkpoint文件夹的路径
    model = AutoModelForSequenceClassification.from_pretrained(params_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(params_path)
    return (model,tokenizer)#predictor1,

    
def predict_fn(input_data, model):
    if input_data[0]=='sagemaker-kwm-new':# model name needs to mapped
        idx2label = [1, 0]
        tokenizer=model[1]
        new_model = model[0]
        inputs = tokenizer([input_data[1]], max_len_seq=512)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        input_ids, token_type_ids = paddle.to_tensor(input_ids), paddle.to_tensor(token_type_ids)
        logits = new_model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        output = [idx2label[i] for i in idx]
    else:
        break
    return output

def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps(prediction, ensure_ascii=False, cls=NpEncoder)
    return prediction

@app.route('/ping', methods=['GET'])
def ping():
    health = 1
    status = 200 if health else 404
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    print("================ INVOCATIONS =================")
    print ("<<<< flask.request.content_type", flask.request.content_type)
    paddle.device.set_device("gpu")
    print(paddle.device.get_device())
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)

        model = model_fn('/opt/program/')
        result = predict_fn(data, model)
        result_json = output_fn(result, content_type='application/json')

        return flask.Response(response=result_json, status=200, mimetype='application/json')
