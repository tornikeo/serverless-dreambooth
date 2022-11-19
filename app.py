import os
import json
# import torch
import zipfile
# import googledrive as gd
from zipfile import ZipFile
# from transformers import pipeline
# from diffusers.models import AutoencoderKL
# from diffusers import StableDiffusionPipeline
import boto3
from pathlib import Path
import shutil
import tempfile
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    pass
    # global vae
    # model = "runwayml/stable-diffusion-v1-5"
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    
    # global pipe
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     model, 
    #     vae=vae, 
    #     torch_dtype=torch.float16, 
    #     revision="fp16"
    # ).to("cuda")
    # print("done")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    # global model
    # global vae

    # Parse out arguments
    zip_file_id = model_inputs.get('file_id', None)
    BUCKET_NAME = "tornikeo-ml-model-files"
    #From parameter file_id download the zip from UploadedPics folder
    # zip_file_name = gd.download(zip_file_id)

    s3 = boto3.client('s3')
    obj_name = zip_file_id.split('/')[-1]
    # obj_name = upload_dir('/home/tornikeo/Downloads/db_inp_ani', BUCKET_NAME)
    with tempfile.TemporaryDirectory() as dir_name:
        dir_name = Path(dir_name)
        download_file_name = Path(dir_name) / obj_name
        s3.download_file(BUCKET_NAME, obj_name, str(download_file_name))
        unpack_dir = Path(dir_name) / 'data/sks'
        shutil.unpack_archive(str(download_file_name), unpack_dir)
        # Setup concepts_list
        concepts_list = [
            {
                "instance_prompt":      "photo of sks person",
                "class_prompt":         "photo of a person",
                "instance_data_dir":    str(unpack_dir),
                "class_data_dir":       str("cached/person")
            }
        ]
        # 'class_data_dir' contains regularization images
        # 'instance_data_dir' is where training images go
        for c in concepts_list:
            os.makedirs(c["instance_data_dir"], exist_ok=True)

        #Unzip training images
        # train_path = concepts_list[0]["instance_data_dir"]
        # with zipfile.ZipFile('sks.zip', 'r') as f:
        #     f.extractall('data/sks')
        # Create concept file

        concepts_list_path = str(dir_name / 'concepts_list.json')
        with open(concepts_list_path, "w") as f:
            json.dump(concepts_list, f, indent=4)
        
        #Call training script
        output_dir = str(dir_name / 'stable_diffusion_weights')
        model_name = Path('./model_weights')
        train = os.system(f"OUTPUT_DIR={output_dir} CONCEPTS_LIST_PATH={concepts_list_path} MODEL_NAME={model_name} bash train.sh")
        print(train)

        #Compressed model to half size (4Gb -> 2Gb) to save space in gdrive folder: Models/
        steps = 1200
        compress = os.system("python convert_diffusers_to_original_stable_diffusion.py --model_path 'stable_diffusion_weights/"+str(steps)+"/' --checkpoint_path ./model.ckpt --half")
        print(compress)

        #Upload model.ckpt file to gdrive Folder: Models/
        newId = gd.upload('model.ckpt')

        # Return the results as a dictionary
        return {'response': str(newId)}