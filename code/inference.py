# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import io
import json
import logging
import os
import tempfile

import cv2
import torch
import torchvision.transforms as transforms

# This code will be loaded on each worker separately..
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    device = get_device()
    logger.info(">>> Device is '%s'.." % device)
    model = torch.load(model_dir + '/model.pth', map_location=torch.device(device))
    print(type(model))
    logger.info(">>> Model loaded!..")
    return model

def transform_fn(model, request_body, content_type, accept):
    interval = int(os.environ.get('FRAME_INTERVAL', 30))
    frame_width = int(os.environ.get('FRAME_WIDTH', 1024))
    frame_height = int(os.environ.get('FRAME_HEIGHT', 1024))
    batch_size = int(os.environ.get('BATCH_SIZE', 24))

    f = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    all_predictions = []

    for batch_frames in batch_generator(tfile, frame_width, frame_height, interval, batch_size):
        batch_inputs = preprocess(batch_frames)  # returns 4D tensor
        batch_outputs = predict(batch_inputs, model)
        logger.info(">>> Length of batch predictions: %d" % len(batch_outputs))
        batch_predictions = postprocess(batch_outputs)
        all_predictions.extend(batch_predictions)
    
    logger.info(">>> Length of final predictions: %d" % len(all_predictions))
    return json.dumps(all_predictions)

def preprocess(inputs, preprocessor=transforms.ToTensor()):
    outputs = torch.stack([preprocessor(frame) for frame in inputs])
    return outputs
    
def predict(inputs, model):
    logger.info(">>> Invoking model!..")

    with torch.no_grad():
        device = get_device()
        model = model.to(device)
        input_data = inputs.to(device)
        model.eval()
        outputs = model(input_data)

    return outputs

def postprocess(inputs):
    outputs = []
    for inp in inputs:
        outputs.append({
            'boxes': inp['boxes'].detach().cpu().numpy().tolist(),
            'labels': inp['labels'].detach().cpu().numpy().tolist(),
            'scores': inp['scores'].detach().cpu().numpy().tolist()
        })
    return outputs

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def batch_generator(tfile, frame_width, frame_height, interval, batch_size):
    cap = cv2.VideoCapture(tfile.name)
    frame_index = 0
    frame_buffer = []

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            cap.release()
            if frame_buffer:
                yield frame_buffer
            return

        if frame_index % interval == 0:
            frame_resized = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            frame_buffer.append(frame_resized)

        if len(frame_buffer) == batch_size:
            yield frame_buffer
            frame_buffer.clear()

        frame_index += 1
    else:
        raise Exception("Failed to open video '%s'!.." % tfile.name)


