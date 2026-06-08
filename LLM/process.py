import json
import pandas as pd
import PIL
from PIL import Image
import tqdm
import pickle
import os

def load_data(file_path: str ='../deeplearning/raw_data/filtered_photos.json') :
    with open(file_path,'r') as f :
        data = json.load(f)

    df = pd.DataFrame(data)
    return df

def process_data(df: pd.DataFrame) :
    """
    1. 处理文件名，加上文件夹相对路径
    2. 处理受损图片
    2. 处理label
    3. 给caption编码
    max_len: 最大长度
    min_freq: 被保留词语最小词频
    """
    print(">>> Reading Data")
    df['image_path'] = '../deeplearning/raw_data/yelp_filtered_image/' + df['photo_id'].astype(str) + '.jpg'

    broken_images_file = "broken_images.pkl"
    if os.path.exists(broken_images_file):
        print(">>> Found existing broken_images list, loading...")
        with open(broken_images_file, "rb") as f:
            broken_images = pickle.load(f)
    else:
        count = 0
        broken_images = []
        for path in tqdm.tqdm(df['image_path']):
            try:
                Image.open(path).verify()
            except (PIL.UnidentifiedImageError, OSError):
                count += 1
                broken_images.append(path)
        print(f">>> {count} photos cannot be opened and removed")
        with open(broken_images_file, "wb") as f:
            pickle.dump(broken_images, f)
    df = df[~df['image_path'].isin(broken_images)].copy()
    print(f">>> Removed {len(broken_images)} broken images")

    df_no_caption = df[df['caption']==""].reset_index().copy()
    df_with_caption = df[~(df['caption']=="")].reset_index().copy()

    print(f">>> {len(df_with_caption)} photos have a caption.")
    print(f">>> {len(df_no_caption)} photos do not have a caption.")

    return df_no_caption, df_with_caption