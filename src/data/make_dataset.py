from asyncio.tasks import gather
from typing import Optional
import datetime as dt
import gc

from distributed import client

from src.data.utils import get_dask_df, process_minute_level, process_minute_level_pandas
from src.utils import get_logger
logger = get_logger(__name__)

import click

import dask.dataframe as dd
import dask
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

import numpy as np
import pandas as pd
import time

# from memory_profiler import profile


# def str_col_to_list(item: str, default_len: int = 24*60,
#                     sep_char: str = " ",dtype=int):
#     if isinstance(item,str):
#         return item.split(sep_char)
#     else: 
#         return [np.nan] * default_len


def explode_str_column_dask(df: dd, target_col: str,
                   freq: str = "min", dur: str = "1D",
                   sep_char: str = " ", date_col: str = "dt",
                   dtype: str  = "Int32", participant_col: str = "id_participant_external",
                   rename_participant_id_column: str = "participant_id",
                   rename_target_column: Optional[str] = None,
                   start_col: Optional[str] = None,
                   dur_col: Optional[str] = None) -> dict:
    
    df["timestamp"] = df.apply(get_new_index,target_column = target_col,
                                    start_col=start_col,
                                    dur_col=dur_col,
                                    freq=freq,
                                    dur=dur,
                                    axis=1,
                                    date_col=date_col,
                                    meta="datetime64")


    # mapper = lambda x: str_col_to_list(x, sep_char=sep_char, default_len=default_len)                                        
    df["val"] = df[target_col].str.split(sep_char)
    print(df)
    df = df[["timestamp","val"]].explode(["timestamp","val"])
    # timestamps = df["timestamp"].compute()
    # pd_df = df.compute()
    # # print(timestamps.value_counts().sort_values(ascending=False))
    # # assert  timestamps.is_unique
    # assert len(pd_df.drop_duplicates(subset=["timestamp",participant_col])) == len(pd_df)
    # Need to convert to float first to handle NaN: 
    # https://stackoverflow.com/questions/39173813/pandas-convert-dtype-object-to-int
    df["val"] = dd.to_numeric(df["val"]).astype(dtype)
    
    val_col_name = rename_target_column if rename_target_column else target_col

    df = df.rename(columns={"val":val_col_name})
    df.index = df.index.rename(rename_participant_id_column)
    return df

def explode_str_column(df: dd, target_col: str,
                   freq: str = "min", dur: str = "1D",
                   sep_char: str = " ", date_col: str = "dt",
                   dtype: str  = "Int32", participant_col: str = "id_participant_external",
                   rename_participant_id_column: str = "participant_id",
                   rename_target_column: Optional[str] = None,
                   start_col: Optional[str] = None,
                   dur_col: Optional[str] = None) -> dict:

    tqdm.pandas(desc="Getting new indices...")
    df["timestamp"] = df.progress_apply(get_new_index,target_column = target_col,
                                    start_col=start_col,
                                    dur_col=dur_col,
                                    freq=freq,
                                    dur=dur,
                                    axis=1,
                                    date_col=date_col)


    # mapper = lambda x: str_col_to_list(x, sep_char=sep_char, default_len=default_len)                                        
    df["val"] = df[target_col].str.split(sep_char)
    df = df[[participant_col,"timestamp","val"]].explode(["timestamp","val"])
    # timestamps = df["timestamp"].compute()
    # pd_df = df.compute()
    # # print(timestamps.value_counts().sort_values(ascending=False))
    # # assert  timestamps.is_unique
    # assert len(pd_df.drop_duplicates(subset=["timestamp",participant_col])) == len(pd_df)
    # Need to convert to float first to handle NaN: 
    # https://stackoverflow.com/questions/39173813/pandas-convert-dtype-object-to-int
    df["val"] = pd.to_numeric(df["val"],meta=dtype,downcast="ignore")
    
    val_col_name = rename_target_column if rename_target_column else target_col
    pid_col_name = rename_participant_id_column if rename_participant_id_column else participant_col

    df = df.rename(columns={"val":val_col_name,
                            participant_col:pid_col_name})
    return df
    

def get_new_index(item: dict, target_column: str,
                   freq: str = "min", dur: str = "1D",
                   sep_char: str = " ", date_col: str = "dt",
                   start_col: Optional[str] = None,
                   dur_col: Optional[str] = None) -> list:

    if start_col:
        start = item[start_col]
        start_ts = pd.to_datetime(start).round(freq)
        end_ts = start_ts + pd.to_timedelta(item[dur_col],unit=freq)
    else:    
        start_ts = pd.to_datetime(item[date_col])
        end_ts = start_ts + pd.to_timedelta(dur)

    new_index = pd.date_range(start_ts,end_ts,freq=freq,closed="left")
    return new_index

CHUNKSIZE="1GB"
PARTITION_SIZE="1GB"
def read_raw_dask(path):
        df = dd.from_pandas(read_raw_pandas(path),npartitions=1024)#.set_index("id_participant_external")
        # dd.read_parquet(path,
        #                     engine='pyarrow-legacy', 
        #                     index="id_participant_external").dropna()
        return df 

def read_raw_pandas(path,set_dtypes=None):
    df = pd.read_parquet(path,engine='pyarrow').set_index("id_participant_external")
    # df["id_participant_external"] = df["id_participant_external"].astype("category")
    if set_dtypes:
        for k,v in set_dtypes.items():
            df[k] = df[k].astype(v)
            
    return df.dropna()


@click.command()
@click.argument("sleep_in_path", type=click.Path(exists=True))
@click.argument("steps_in_path", type=click.Path(exists=True))
@click.argument("heart_rate_in_path",type=click.Path(exists=True))
@click.argument("out_path",type=click.Path())
def main(sleep_in_path: str, steps_in_path: str, 
         heart_rate_in_path: str, out_path: str) -> None:
    
    dask.config.set(scheduler='single-threaded')
    client = Client(n_workers=8)
    logger.info("Loading sleep...")
    exploded_sleep = explode_str_column_dask(read_raw_dask(sleep_in_path),
                                    target_col = "minute_level_str",
                                    rename_target_column="sleep_classic",
                                    start_col="main_start_time",
                                    dur_col = "main_in_bed_minutes",
                                    dtype=pd.Int64Dtype())#.set_index(["participant_id","timestamp"])
    # exploded_sleep["sleep_classic"] = exploded_sleep["sleep_classic"].astype("category")
    keys = ["participant_id","timestamp"]
    logger.info("Loading heart rate...") 
    exploded_hr = explode_str_column_dask(read_raw_dask(heart_rate_in_path),
                                target_col = "minute_level_str",
                                rename_target_column="heart_rate",
                                dtype=pd.Int64Dtype())


    sleep_and_hr = exploded_sleep.merge(exploded_hr,
                                        left_on = keys,
                                        right_on = keys,
                                        how = "outer")                                

    logger.info("Loading steps...") 
    exploded_steps = explode_str_column_dask(read_raw_dask(steps_in_path),
                                target_col = "minute_level_str",
                                rename_target_column="steps",
                                dtype=pd.Int64Dtype())#.set_index(["participant_id","timestamp"])
                                    
   
    # dask.compute(exploded_hr,exploded_sleep,exploded_steps)
    logger.info("Merging...")
   
    all_streams = sleep_and_hr.merge(exploded_steps,
                                    left_on = keys,
                                    right_on = keys,
                                    how = "outer")
    # client.run(gc.collect)

    # all_streams = pd.concat([exploded_hr,exploded_steps,exploded_sleep],axis=1)
    # all_streams["participant_id"] = all_streams["participant_id"].astype("category")
    process_minute_level(minute_level_df=all_streams, out_path=out_path)

if __name__ == "__main__":
    main()