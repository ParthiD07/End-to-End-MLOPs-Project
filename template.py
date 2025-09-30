import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

project_name="Easy_Visa"

list_of_files=[
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception.py",
    f"{project_name}/logging/__init__.py",
    f"{project_name}/logging/logger.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename = os.path.split(filepath)

    # --- Directory Creation Logic ---
    # Determine if directory was created or already existed
    is_dir_newly_created = False
    if filedir and not os.path.exists(filedir):
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir}")
        is_dir_newly_created = True

    # Log 'already exists' for the directory only if it wasn't newly created
    if filedir and not is_dir_newly_created and os.path.exists(filedir):
        logging.info(f"Directory already exists: {filedir}")

    # --- File Creation Logic ---
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) ==0):
        with open(filepath,"w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else: # This else block is only reached if the file ALREADY existed and was not empty
        logging.info(f"File already exists: {filepath}")

    


