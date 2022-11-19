# This file is used to locally verify your http server acts as expected
# Run it as: 'python3 test.py (file_id of training images in google drive)'

import sys
import requests
from io import BytesIO
from PIL import Image
import time

while True:
    try:
        resp = requests.post('http://localhost:8000', json={'file_id':'tornikeo-ml-model-files/db_inp_anidbd30cba.zip'})
    except Exception as e:
        print(e)
        time.sleep(.5)

# #File_id of file in google drive
# file_id = sys.argv[1:][0]

# #Serverless approach: Just need fileId to download training images from folder UploadedImgs/
# model_inputs = {"file_id": file_id }
# res = requests.post('http://localhost:8000/', json = model_inputs)

# #Get back file_id of the newly created model.ckpt stored now in googledrive folder Models/
# print(res.json())


# import torch
# from diffusers import StableDiffusionPipeline
# from diffusers.models import AutoencoderKL
# from huggingface_hub import HfFolder
# from transformers import pipeline

# pipe = StableDiffusionPipeline.from_pretrained(
#     'model_weights', 
#     torch_dtype=torch.float16, 
#     revision="fp16"
# ).to("cuda")
# print(pipe)

# import os
# import json
# import torch
# import zipfile
# import googledrive as gd
# from zipfile import ZipFile
# from transformers import pipeline
# from diffusers.models import AutoencoderKL
# from diffusers import StableDiffusionPipeline

# # zip_file_name = gd.download(zip_file_id)
# newId = gd.upload('model.ckpt')
# print(newId)
