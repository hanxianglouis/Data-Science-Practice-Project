import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
import datetime
import json

from classification_utils import *
from model_utils import load_model_and_processor

def main(image=True, text=True) :
    if (not image) and (not text) :
        raise ValueError("At least one modality should be provided.")

    df = pd.read_csv("../deeplearning/raw_data/caption_completed.csv")
    df = df.sort_values(by="photo_id").reset_index(drop=True) # Guarantee that the order of samples is the same as that used in the Deep Learning model.

    df_train, df_temp = train_test_split(df,test_size=0.3,random_state=42,shuffle=True)
    df_valid, df_test = train_test_split(df_temp,test_size=2/3,random_state=42,shuffle=True)

    model, processor = load_model_and_processor("./qwen_model_cache/Qwen2.5-VL-3B-Instruct")
    model.to("cuda")
    print(">>> Model moved to GPU")

    pred_labels = []

    if image and text :
        for image_path, caption in tqdm.tqdm(zip(df_test['image_path'].tolist(), df_test['caption'].tolist()), total=len(df_test)) :
            pred_label = classify_image_with_caption(model=model, processor=processor, image_path=image_path, caption=caption)
            pred_labels.append(pred_label)

    elif image and (not text) :
        for image_path in tqdm.tqdm(df_test['image_path'].tolist()) :
            pred_label = classify_image_with_caption(model=model, processor=processor, image_path=image_path)
            pred_labels.append(pred_label)

    elif text and (not image) :
        for caption in tqdm.tqdm(df_test['caption'].tolist()) :
            pred_label = classify_image_with_caption(model=model, processor=processor, caption=caption)
            pred_labels.append(pred_label)

    df_test[f'llm_pred_image={image}_text={text}'] = pred_labels

    # Generate timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df_test.to_csv(f"LLM_classification_test_result_{ts}.csv", index=False)

    test_metrics = calc_metrics_llm(true_label=df_test['label'].tolist(), pred_label=pred_labels)

    print("=== Test Set Metrics ===")
    print(test_metrics)

    # Save Path (Customizable)
    save_dir = "../deeplearning/results"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"LLM_image={image}_text={text}_{ts}.json")

    with open(save_path, "w") as f:
        json.dump(test_metrics, fp=f, indent=4)

    print(f"Test results have been saved to: {save_path}")


if __name__ == "__main__" :
    main(image=True, text=True)
    main(image=True, text=False)
    main(image=False, text=True)