# About
This repository include code for our project of the course Data Science Practice in Renmin University of China. The main task is to categorize photos with captions, which is from [Yelp](https://www.yelp.com/dataset). We will attempt to use the photos only, use the captions (text) only, and use both of them simultaneously. As this is just a course programe, we do not guarantee any commercial value of the method we propose, and everyone is welcome to refer to my method or improve such method.

According to the task description, we have three problems to research on:

1. Split the dataset into training, validation, and test sets in a 7:1:2 ratio. Perform some data visualization on the training set. Attempt five-class classification using only text and using only images, and report the Precision, Recall, and F1 scores for each category. Report the macro-F1 score.
2. Combine text and images to perform a five-class classification task. Compared to using only one modality, does the model show improved performance?
3. What should one do when using an LLM? Is it better than non-LLM methods? Or are methods based on LLMs better? Why is that the case?

# Data

As mentioned in the official website of Yelp's public data, there are 200,000 photos in total. However, our teacher just provided us 20,000 of them and we can only test on the data provided.

During experiments, we find that here are some damaged photos, and the code can eliminate them automatically.

Attention, the data about photos' name, captions and labels, which is named as `filtered_photo.json`, should be positioned at path `./deeplearning/raw_data/filtered_photo.json`. And all the photos should be put at folder `./deeplearning/raw_data/yelp_filtered_image/`.

Of course, you can use your own data as long as it includes the photos, captions and labels. But you may have to change the process code in `./deeplearning/load_and_process.py`.

# Run the code
At the beginning, use
```bash
pip install -r requirements.txt
```
to establish the environment.
## Deep Learning model
First, use
```bash
cd /deeplearning/
```
to ender the folder `deeplearning`. Then you can run `run.py` directly with
```bash
python run.py
```
The code will create some folders automatically and output the test metrics which will be stored in folder `./results/` with JSON format.

At the beginning of the progress, something indicating the device you are using will show in the terminal, which can be

```bash
Using device: cpu
```

or

```bash
Using device: cuda
```

or

```bash
Using device: mps
```

Because the model is kind of complex, we do not recommend you to run the code on a computer that can only use CPU as it is inefficient. CUDA is the most recommended environment to run, while MPS (i.e. Mac with Apple Silicon) is acceptable.

We have to acknowledge that, during computing, most of the time is consumed on I/O process that read the photos because it is memory-wasting to read all photos once. To accelerate, you can adjust the `num_workers` parameter in train, valid and test DataLoader, which can be found in `train.py` and `evaluate.py`. We must point out that the optimal number of workers is not fixed but depends on your device. You can try 2, 4, 8, 16 and choose the best one.

## LLM-based operation
First, use
```bash
cd /LLM/
```
to ender the folder `deeplearning`. Then, run the script `adapt.py` to execute the LoRA adaptation. After that, run `complete.py` to complete the empty captions with the adapted LLM. Finally, run `classify.py` to test the LLM's ability of classifying the photos and captions, and the results will be stored in `../deeplearning/results`.

We choose the [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) as the pre-trained LLM for our experiments. You can choose any other models but you may have to change the structure to match other models. Please note that LLM inference is highly demanding in terms of computing power and VRAM; ensure that your device is capable of supporting the inference and fine-tuning of large models. For us, all the experiments are executed on a workstation with an INTEL(R) XEON(R) GOLD 6530 CPU (62GB), and an NVIDIA GeForce RTX 4090 GPU (24GB). Attention, you may find that the LLM-related code somehow support MPS devices (Mac with Apple Silicon), however it can only support the inference process not the adaptation process due to software limitations.