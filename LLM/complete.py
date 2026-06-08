import tqdm
import pandas as pd
import os

from model_utils import generate_caption, load_model_and_processor
from process import load_data, process_data

def complete_caption(df_no_caption, model, processor) :
    print(">>> Begin completing the caption using the adatped model.")
    captions = []
    for image_path in tqdm.tqdm(df_no_caption['image_path']) :
        caption = generate_caption(model=model, processor=processor, image_path=image_path)
        captions.append(caption)

    df_no_caption['caption'] = captions

    return df_no_caption

def main():
    df = load_data()
    df_no_caption, df_with_caption = process_data(df)

    model, processor = load_model_and_processor("./qwen_model_cache/qwen2_5_vl_3b_caption_lora")
    model.to("cuda")
    print(">>> Model moved to GPU")
    
    df_no_caption_completed = complete_caption(df_no_caption=df_no_caption, model=model, processor=processor)

    df_completed = pd.concat([df_no_caption_completed, df_with_caption], axis=0)
    df_completed = df_completed.sample(frac=1, random_state=42).reset_index(drop=True)

    save_path = "../deeplearning/raw_data/"
    os.makedirs(save_path, exist_ok=True)

    df_completed.to_csv(save_path + "caption_completed.csv", index=False)

if __name__ == "__main__" :
    main()