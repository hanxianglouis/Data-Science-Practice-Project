import os
import re
import torch
from qwen_vl_utils import process_vision_info
from sklearn.metrics import precision_score, f1_score, precision_recall_fscore_support, accuracy_score
import numpy as np

class Prompt() :

    def __init__(self):

        self.only_image = """
You are given an image. 
Your task is to classify the image into exactly one of the following five categories:

inside, outside, food, menu, drink

Definitions:
- inside: indoor restaurant/cafe environment, interior space, tables, chairs, counter, decoration
- outside: exterior view, storefront, street view, outdoor scene
- food: edible food items, dishes, meals, desserts, snacks
- menu: menu board, printed menu, price list, ordering screen
- drink: beverages, cups, coffee, tea, juice, alcohol-free drinks

Important:
Only output one word from this list:
inside, outside, food, menu, drink
Do not explain your answer.
"""
        self.only_text = """
You are given a caption of an image. 
Your task is to classify the image into exactly one of the following five categories relying solely on its caption:

inside, outside, food, menu, drink

Definitions:
- inside: indoor restaurant/cafe environment, interior space, tables, chairs, counter, decoration
- outside: exterior view, storefront, street view, outdoor scene
- food: edible food items, dishes, meals, desserts, snacks
- menu: menu board, printed menu, price list, ordering screen
- drink: beverages, cups, coffee, tea, juice, alcohol-free drinks

Caption:
{caption}

Important:
Only output one word from this list:
inside, outside, food, menu, drink
Do not explain your answer.
"""
        self.image_and_text = """
You are given an image and its caption. 
Your task is to classify the image into exactly one of the following five categories:

inside, outside, food, menu, drink

Definitions:
- inside: indoor restaurant/cafe environment, interior space, tables, chairs, counter, decoration
- outside: exterior view, storefront, street view, outdoor scene
- food: edible food items, dishes, meals, desserts, snacks
- menu: menu board, printed menu, price list, ordering screen
- drink: beverages, cups, coffee, tea, juice, alcohol-free drinks

Caption:
{caption}

Important:
Only output one word from this list:
inside, outside, food, menu, drink
Do not explain your answer.
"""



@torch.no_grad()
def classify_image_with_caption(model, processor, image_path=None, caption=None, max_new_tokens=16):
    if (image_path is None) and (caption is None) :
        raise ValueError("At least one modality should be provided.")
    
    prompts = Prompt()
    if (image_path is None) and (caption is not None) :
        prompt = prompts.only_text.format(caption=caption)
    elif (image_path is not None) and (caption is None) :
        prompt = prompts.only_image
    else :
        prompt = prompts.image_and_text.format(caption=caption)

    if image_path is None : # When there is only text information, the format is different
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    else :

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if image_path is None :
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

    else :

        image_inputs, _ = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Clean the output and keep the class only.
    output_lower = output_text.strip().lower()

    valid_labels = ["inside", "outside", "food", "menu", "drink"]
    for label in valid_labels:
        if re.search(rf"\b{label}\b", output_lower):
            return label

    # If no valid tags are found, the original output is returned to help debug.
    return output_text.strip()

def calc_metrics_llm(true_label: list[str],pred_label: list[str]):

    if len(true_label) != len(pred_label):
        raise ValueError(
            f"true_label and pred_label must have the same length, "
            f"but got {len(true_label)} and {len(pred_label)}."
        )
    
    labels = ["inside", "outside", "food", "menu", "drink"]

    precision, recall, f1, support = precision_recall_fscore_support(
        true_label,
        pred_label,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    result = {
        "per_class": per_class,
        "macro_f1": float(f1_score(true_label, pred_label, labels=labels, average="macro", zero_division=0)),
    }

    return result