import os
import yaml
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from box.exceptions import BoxValueError
from typing import Any,Union
import numpy as np
import dill
from pandas import DataFrame
from Easy_Visa.logging.logger import logger
from Easy_Visa.exception.exception import CustomException

def read_yaml(path_to_yaml: Union[str, Path])-> dict:
    """
    Reads a YAML file and returns its contents as a dictionary-like object.

    Args:
        path_to_yaml (Path): Path object or string pointing to the YAML file.

    Raises:
        ValueError: If YAML file is empty.
        CustomException: If any other exception occurs while reading.

    Returns:
        dict: Parsed YAML contents.
    """
    try:
        with open(path_to_yaml,"r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully")
            return content
  
    except Exception as e:
        raise CustomException(e)    


def save_yaml(path_to_yaml: Union[str, Path],data: dict) -> None:
    """
    Saves a dictionary to a YAML file.

    Args:
        path_to_yaml (Path): Path object or string where YAML file will be saved.
        data (dict): Dictionary data to write into the YAML file.

    Raises:
        CustomException: If writing fails due to I/O or other issues.
    """
    try: 
        os.makedirs(os.path.dirname(path_to_yaml),exist_ok=True)
        with open(path_to_yaml,"w") as yaml_file:
            yaml.safe_dump(data,yaml_file)
        
        logger.info(f"YAML file saved successfully at '{path_to_yaml}'")

    except Exception as e:
        raise CustomException(e)

def save_object(file_path: Union[str, Path],model:Any) -> None:
    """
    Save Python object (e.g., ML model) to a file using dill.

    Args:
        obj (Any): Object to be serialized and saved.
        path (Path): File path where the object will be stored.
    """
    try:
        file_path = Path(file_path) 
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(file_path,"wb") as f:
            dill.dump(model,f)

        logger.info(f"Object saved successfully at: {file_path}")
    
    except Exception as e:
        raise CustomException(e)
    
def load_object(file_path: Union[str, Path]) -> Any:
    """
    Load Python object (e.g., ML model) from a dill file.

    Args:
        path (Path): Path to the saved dill file.

    Returns:
        Any: Loaded Python object.
    """
    try:
        with open (file_path, "rb") as f:
            obj = dill.load(f)
        logger.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e)
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """Create multiple directories if they don't exist.

    Args:
        path_to_directories(list): list of path of directories
        verbose (bool, optional): If True, log each directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """
    Save a NumPy array to a file in .npy format.

    Args:
        file_path (str): Path to save the array.
        array (np.ndarray): The NumPy array to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise CustomException(e)
    
def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Load a NumPy array from a .npy file.

    Args:
        file_path (str): Path of the saved array file.

    Returns:
        np.ndarray: The loaded NumPy array.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e)
    



def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop specified columns from a pandas DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.
        cols (list): List of column names to drop.

    Returns:
        DataFrame: DataFrame after dropping the specified columns.
    """
    try:
        # Drop given columns
        updated_df = df.drop(columns=cols, axis=1)

        logger.info(f"Dropped columns: {cols}")

        return updated_df

    except KeyError as e:
        logger.error(f"One or more columns not found in DataFrame: {cols}")
        raise CustomException(e)
    except Exception as e:
        raise CustomException(e)