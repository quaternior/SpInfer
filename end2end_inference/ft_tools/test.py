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

# 打印包的版本信息
print(f"argparse version: {argparse.__version__}")
# print(f"configparser version: {configparser.__version__}")
# print(f"multiprocessing version: {multiprocessing.__version__}")
print(f"numpy version: {np.__version__}")
# print(f"pathlib version: {Path.__version__}")  # Path 没有 __version__ 属性
print(f"torch version: {torch.__version__}")
# print(f"os version: {os.__version__}")  # os 没有 __version__ 属性
# print(f"sys version: {sys.__version__}")  # sys 没有 __version__ 属性
# print(f"datetime version: {datetime.__version__}")  # datetime 没有 __version__ 属性
print(f"transformers version: {importlib.metadata.version('transformers')}")

# 检查 OPTForCausalLM 和 AutoModelForCausalLM 是否存在
try:
        from transformers import OPTForCausalLM, AutoModelForCausalLM
        print("OPTForCausalLM and AutoModelForCausalLM are available.")
except ImportError:
        print("OPTForCausalLM and AutoModelForCausalLM are not available.")
