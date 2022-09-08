import os
import cv2
import time
import threading
import queue
import argparse
import json

import towhee
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
import redis


from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
import torch

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)
q = queue.Queue(1)

connections.connect(host="milvus-standalone", port=19530)
red = redis.Redis(host='redis', port=6379, db=0)
red.flushdb()
collection_name = "image_similarity_search"
dim = 2048
default_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
default_schema = CollectionSchema(fields=default_fields, description="Image test collection")
collection = Collection(name=collection_name, schema=default_schema)
default_index = {"index_type": "IVF_SQ8", "params": {"nlist": 512}, "metric_type": "L2"}
collection.create_index(field_name="vector", index_params=default_index)
collection.load()

@app.route('/submit/image', methods=['POST'])
def submit_image():
    json_data = json.loads(request.data)

    camera_id = json_data['camera_id']
    filename = json_data['filename']
    print(f'filename: {filename}, camera_id: {camera_id}')

    try:
        q.put_nowait(filename)
        return 'processing', 200
    except:
        os.remove(filename)
        return 'skip frame', 200

def worker():
    f = 0
    while True:        
        item = q.get()

        print(f'Working on {item}')
        #  paths = towhee.glob(item).to_list()
        # vectors = towhee.glob(item).exception_safe() \
        #                 .image_decode() \
        #                 .image_embedding.timm(model_name="resnet50") \
        #                 .drop_empty() \
        #                 .tensor_normalize() \
        #                 .to_list()

        # Initialize Img2Vec with GPU
        img2vec = Img2Vec(cuda=False)

        # Read in an image (rgb format)
        img = Image.open(item)
        # Get a vector from img2vec, returned as a torch FloatTensor
        vec = img2vec.get_vec(img, tensor=True)
        print(vec)
        
        mr = collection.insert([vec])
        ids = mr.primary_keys
        print(ids)
        #for x in range(len(ids)):
        #    red.set(str(ids[x]), paths[x])
    
        print(f'Finished {item}')
        q.task_done()
    

# Turn-on the worker thread.
threading.Thread(target=worker, daemon=True).start()

app.run(host='0.0.0.0', port=3000)
