from qwen_vl_utils import process_vision_info

class QwenVLCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        image_inputs_list = []
        video_inputs_list = []

        for ex in examples:
            messages = ex["messages"]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            texts.append(text)
            image_inputs_list.append(image_inputs)

        batch = self.processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        # pad token not in loss
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # image token not in loss
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_token_id is not None:
            labels[labels == image_token_id] = -100


        batch["labels"] = labels

        return batch