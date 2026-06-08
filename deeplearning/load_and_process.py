import json
import pandas as pd
import re
from collections import Counter
import PIL
from PIL import Image
import tqdm
import pickle
import os

def load_data(file_path: str ='./raw_data/filtered_photos.json', completed: bool =False) :
    if completed :
        df = pd.read_csv("./raw_data/caption_completed.csv")
    else :
        with open(file_path,'r') as f :
            data = json.load(f)

        df = pd.DataFrame(data)
    return df

def process_data(df: pd.DataFrame, max_len: int =10, min_freq: int =3, completed: bool =False) :
    """
    1. 处理文件名，加上文件夹相对路径
    2. 处理受损图片
    2. 处理label
    3. 给caption编码
    max_len: 最大长度
    min_freq: 被保留词语最小词频
    """
    print(">>> Reading Data")
    df['image_path'] = './raw_data/yelp_filtered_image/' + df['photo_id'].astype(str) + '.jpg'

    if not completed : # When using the raw data without using LLMs to complete caption
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

    labels = sorted(df['label'].dropna().unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df['label_id'] = df['label'].map(label2id)
    df['tokenized_id_context'], vocab_size = tokenize_and_encode(df['caption'], max_len=max_len, min_freq=min_freq)

    df = df.sort_values(by="photo_id").reset_index(drop=True) # Guarantee that the order of samples in the dataframe keep the same when we use the raw data and the caption-completed data. Therefore, the train set, valid set and test set will be exactly the same. 

    print(f">>> Sample size after processing: {len(df)}")

    return df, label2id, id2label, vocab_size

def tokenize(text, max_len=10):
    """
    英文分词器：
    - 转小写
    - 保留单词和常见标点
    """
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+|[.,!?;:()\-\']", text)
    return tokens[:max_len] # 只保留到最大长度，避免构建word2id的时候保留了被去掉的词

def tokenize_and_encode(contexts: list[str], max_len: int =10, min_freq: int =3) -> tuple[list[list[int]], int] :
    """
    将原文本分词并转换为id，并完成padding
    max_len: 最大长度
    min_freq: 被保留词语最小词频
    """
    print(">>> Tokenizing")
    tokenized_context = [tokenize(text, max_len=max_len) for text in contexts]
    counter = Counter()
    for tokens in tokenized_context:
        counter.update(tokens)

    word2id = {
        "<pad>": 0,
        "<unk>": 1
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            word2id[word] = len(word2id)

    tokens = []
    for word_token in tokenized_context :
        seq = []
        for word in word_token :
            seq.append(word2id.get(word, 1))
        if len(seq) < max_len :
            seq = seq + [word2id["<pad>"]] * (max_len - len(seq))
        tokens.append(seq)

    return tokens, len(word2id)
