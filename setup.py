'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
(or distutils in older Python versions) to define the configuration 
of your project, such as its metadata, dependencies, and more
'''

from setuptools import find_packages # find_packages: Automatically detects all Python packages (folders with __init__.py) in your project. Saves you from manually listing them.
from setuptools import setup # setup: The main function that defines your package metadata and installation instructions.
from typing import List # Used for type hinting — tells readers (and tools like linters) that the function returns a list of strings (List[str]).

HYPHEN_E_DOT="-e ."

# Defines a function that reads dependencies from requirements.txt.
def get_requirements(file_path:str)->List[str]: # The return type is a list of strings, each string being a package name.
    """This function will return list of requirements"""
    # Initializes an empty list to store package requirements.

    try:
        with open(file_path,"r") as file:
            # Read lines from file
            requirements_list =[req.strip() for req in file.readlines() if req.strip()]

            if HYPHEN_E_DOT in requirements_list: # Check and remove the hyphenated entry using the string literal
                    requirements_list.remove(HYPHEN_E_DOT)
            return requirements_list

    except FileNotFoundError:
        print(f"Error: Requirements file not found at: {file_path}")
        return []
    
setup(
    name="Easy_visa", # The name of your package (how it will be installed with pip install network-security if published).
    version="0.0.1",
    author="parthi",
    author_email="parthiband2020@gmail.com",
    packages=find_packages(), # Automatically includes all Python packages (folders with __init__.py) inside your project directory.
    install_requires=get_requirements('requirements.txt'), # Instead of hardcoding them, it calls get_requirements() to read from requirements.txt.
)