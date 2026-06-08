import os
import torch
from huggingface_hub import snapshot_download
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def download_model(model_id="Qwen/Qwen2.5-VL-3B-Instruct", local_model_dir="./qwen_model_cache/Qwen2.5-VL-3B-Instruct"):
    if os.path.exists(local_model_dir) and len(os.listdir(local_model_dir)) > 0:
        print(f">>> Model already exists at: {local_model_dir}")
        return local_model_dir

    print(f">>> Downloading model to: {local_model_dir}")

    snapshot_download(
        repo_id=model_id,
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(">>> Model downloaded.")
    return local_model_dir

def load_model_and_processor(model_dir):
    print(">>> Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    print(">>> Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    print(">>> Model loaded.")
    print("Using device:", model.device)
    return model, processor

@torch.no_grad()
def generate_caption(model, processor, image_path, prompt="Please generate a concise and accurate caption for this image."):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

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
        max_new_tokens=128,
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
    )

    return output_text[0]