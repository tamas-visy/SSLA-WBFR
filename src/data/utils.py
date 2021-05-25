import os
import glob 
import json

import pyarrow.parquet as pq
import pandas as pd
from scipy.special import softmax
import pyarrow as pa
from torch.utils import data

from src.utils import get_logger
import src.data.constants as constants

import multiprocessing.popen_spawn_posix

import dask
dask.config.set({"distributed.comm.timeouts.connect": "60"})

import dask.dataframe as dd

from dotenv import dotenv_values

config = dotenv_values(".env")
logger = get_logger()

DATASET_VERSION="2020-07-15"
RAW_DATA_PATH = os.path.join(config["MAIN_PATH"],"data","raw","audere","data-export",DATASET_VERSION)
PROCESSED_DATA_PATH = os.path.join(config["MAIN_PATH"],"data","processed")
DEBUG_DATA_PATH = os.path.join(config["MAIN_PATH"],"data","debug")

def get_raw_dataset_path(name):
    if name in constants.MTL_NAMES:
        return os.path.join(RAW_DATA_PATH,"mtl",name)        
    elif name in constants.SURVEY_NAMES:
        return os.path.join(RAW_DATA_PATH,"survey",name)
    elif name in constants.ACTIVITY_NAMES:
        return os.path.join(RAW_DATA_PATH,"activity",name)
    else:
        raise ValueError(f"Looked for {name} in {RAW_DATA_PATH}, not found!")

def find_raw_dataset(name):
    path = get_raw_dataset_path(name)
    return pq.ParquetDataset(path)

def get_features_path(name):
    if os.environ.get("DEBUG_DATA"): 
        logger.warning("DEBUG_DATA is set, only loading subset of data")
        data_path = DEBUG_DATA_PATH
    else:
        data_path = PROCESSED_DATA_PATH
    return os.path.join(data_path,"features",name+".csv")

def load_raw_table(name,fmt="df"):
    dataset = find_raw_dataset(name)
    logger.info(f"Reading {name}...")
    if fmt=="df":
        return dataset.read().to_pandas()
    elif fmt=="pq":
        return dataset.read()
    else:
        raise ValueError("Unsupported fmt") 

def get_processed_dataset_path(name):
    if os.environ.get("DEBUG_DATA"): 
        logger.warning("DEBUG_DATA is set, only loading subset of data")
        data_path = DEBUG_DATA_PATH
    else:
        data_path = PROCESSED_DATA_PATH
    if name in constants.PROCESSED_DATASETS:
        return os.path.join(data_path,name+".csv")        
    elif name in constants.PARQUET_DATASETS:
        return os.path.join(data_path,name) 
    else:
        raise ValueError(f"Looked for {name} in {data_path}, not found!")

def get_cached_datareader_path(name):
    if os.environ.get("DEBUG_DATA"): 
        logger.warning("DEBUG_DATA is set, only loading subset of data")
        data_path = DEBUG_DATA_PATH
    else:
        data_path = PROCESSED_DATA_PATH
    print(data_path)
    return os.path.join(data_path,"cached_datareaders",name+".pickle")
        
def find_processed_dataset(name):
    path = get_processed_dataset_path(name)
    if ".csv" in path:
        return pd.read_csv(path)
    elif ".jsonl" in path:
        return pd.read_json(path,lines=True)

def load_processed_table(name,fmt="df"):
    dataset = find_processed_dataset(name)
    for column in dataset.columns:
        if "date" in str(column) or "time" in str(column):
            try:
                dataset[column] = pd.to_datetime(dataset[column])
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                continue
    logger.info(f"Reading {name}...")
    if fmt=="df":
        return dataset
    else:
        raise ValueError("Unsupported fmt") 

def write_pandas_to_parquet(df,path,write_metadata=True,
                            partition_cols=[]):
    table = pa.Table.from_pandas(df, preserve_index=False)
    
    pq.write_to_dataset(table, root_path=path,
                    partition_cols=partition_cols)
    if write_metadata:
        paths = glob.glob(os.path.join(path,"*","*.parquet"))
        dd.io.parquet.create_metadata_file(paths)

# @dask.delayed
def get_dask_df(name,path= None,min_date=None,max_date=None,index=None):
    if not path:
        path = get_processed_dataset_path(name)
    filters = []
    # if min_date:
    #     filters.append(("date",">=",min_date))
    # if max_date:
    #     filters.append(("date","<",max_date))
        
    if filters:
        df = dd.read_parquet(path,filters=filters)
    else: 
        df = dd.read_parquet(path,index=index)
    return df

def load_results(path):
    results = pd.read_json(path,lines=True)
    logits = pd.DataFrame(results["logits"].tolist(), columns=["pos_logit","neg_logit"])
    softmax_results = softmax(logits,axis=1)["neg_logit"].rename("pos_prob")
    return pd.concat([results["label"],logits,softmax_results],axis=1)

def write_dict_to_json(data,path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)