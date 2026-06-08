import os
import torch.distributed as dist
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from process import load_data, process_data
from model_utils import download_model, load_model_and_processor
from lora import LoRA

MODEL_DIR = "./qwen_model_cache/Qwen2.5-VL-3B-Instruct"

def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

def wait_for_everyone():
    if is_dist_initialized():
        dist.barrier()


def main():
    df = load_data()
    df_no_caption, df_with_caption = process_data(df)

    if is_main_process():
        print(">>> df_with_caption:", len(df_with_caption))
        print(">>> df_no_caption:", len(df_no_caption))
        model_dir = download_model()
    else:
        model_dir = MODEL_DIR

    wait_for_everyone()

    model, processor = load_model_and_processor(model_dir)

    LoRA(df=df_with_caption,model=model,processor=processor)

    wait_for_everyone()

if __name__ == "__main__" :
    main()