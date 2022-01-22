import os
import json 
import logging
import gc
import wandb

import dotenv
import yaml
import numpy as np
from PIL import Image
import torch
import pynvml
from torchviz import make_dot
import subprocess
from scipy.special import softmax

from dotenv import dotenv_values
config = dotenv_values(".env")

def load_dotenv():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

def read_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)

def clean_datum_for_serialization(datum):
    for k, v in datum.items():
        if isinstance(v, (np.ndarray, np.generic)):
            datum[k] = v.tolist()
    return datum

def write_jsonl(open_file, data, mode="a"):
    for datum in data:
        clean_datum = clean_datum_for_serialization(datum)
        open_file.write(json.dumps(clean_datum))
        open_file.write("\n")

def read_jsonl(path,line=None):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data
    
def get_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    return logger

def check_for_wandb_run():
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run

def render_network_plot(var,dir,filename="model",params=None):
    graph = make_dot(var,params=params)
    graph.format = "png"
    return graph.render(filename=filename,directory=dir)

def get_gpu_memory(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used

def get_unused_gpus():
    result=subprocess.getoutput("nvidia-smi -q -d PIDS |grep -A4 GPU | grep Processes").split("\n")
    return [str(i) for i in range(len(result)) if "None" in result[i]]

def set_gpus_automatically(n):
    free_devices = get_unused_gpus()
    n_free = len(free_devices)
    if n_free < n:
        raise ValueError(f"Insufficent GPUs available for automatic allocation: {n_free} available, {n} requested.")     
    devices = free_devices[:n]

    logger = get_logger(__name__)
    logger.info(f"Auto setting gpus to: {devices}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)


def visualize_model(model,dir="."):
    """
    Returns the path to an image of a model
    """
    x_dummy = torch.rand(model.input_dim).unsqueeze(0)
    y_dummy = torch.tensor(1).unsqueeze(0)
    pred_dummy = model(inputs_embeds=x_dummy, labels = y_dummy)[0]

    params = dict(model.named_parameters())
    model_img_path = render_network_plot(pred_dummy,dir,params=params)
    return model_img_path

def describe_resident_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensors.append((type(obj), obj.size()))
        except:
            pass
    return tensors


def update_wandb_run(run_id,vals):
    project = config["WANDB_PROJECT"]
    entity = config["WANDB_USERNAME"]
    api = wandb.Api()
    run_url = f"{entity}/{project}/{run_id}"
    run = api.run(run_url)
    for k,v in vals.items():
        update_run(run,k,v)
    run.summary.update()
    return  f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

def update_run(run, k, v):
    if (isinstance(run.summary, wandb.old.summary.Summary) and k not in run.summary):
        run.summary._root_set(run.summary._path, [(k, {})])
    run.summary[k] = v


def upload_pandas_df_to_wandb(run_id,table_name,df):
    with get_historical_run(run_id) as run:
        model_table = wandb.Table(dataframe=df)
        run.log({table_name:model_table})


def get_historical_run(run_id: str):
    """Allows restoring an historical run to a writable state
    """
    return wandb.init(id=run_id, resume='allow')

    
def binary_logits_to_pos_probs(arr,pos_index=-1):
    probs = softmax(arr,axis=1)
    return probs[:,pos_index]
