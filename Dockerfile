# Must use at least Cuda version 11+
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /code/app

# Install git
RUN apt-get update && apt-get install -y git build-essential

# Install python packages
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
RUN pip install -U --pre triton
RUN conda install xformers -c xformers/label/dev

# #Requirements.txt
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Add your huggingface auth key here, define models
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}
ENV MODEL_NAME="runwayml/stable-diffusion-v1-5"
ENV OUTPUT_DIR="stable_diffusion_weights/"

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

#Add training/important script
ADD convert_diffusers_to_original_stable_diffusion.py .
ADD train_dreambooth.py .
ADD train.sh .

# Add your custom app code, init() and inference()
ADD app.py .

CMD python3 -u server.py
