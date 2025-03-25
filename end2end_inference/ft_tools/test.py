import argparse
import configparser
import multiprocessing
import numpy as np
from pathlib import Path
import torch
import os
import sys
from datetime import datetime
from transformers import OPTForCausalLM, AutoModelForCausalLM
import importlib.metadata

# check the version of the packages
print(f"argparse version: {argparse.__version__}")
print(f"numpy version: {np.__version__}")
print(f"torch version: {torch.__version__}")
print(f"transformers version: {importlib.metadata.version('transformers')}")

# check if OPTForCausalLM and AutoModelForCausalLM exist
try:
        from transformers import OPTForCausalLM, AutoModelForCausalLM
        print("OPTForCausalLM and AutoModelForCausalLM are available.")
except ImportError:
        print("OPTForCausalLM and AutoModelForCausalLM are not available.")
