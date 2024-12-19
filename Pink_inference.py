import argparse
import os
import json
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import torch
import io
from PIL import Image, ImageDraw
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script with configurable paths.")
    parser.add_argument("--json_folder", type=str, required=True, help="Path to the folder containing JSON files.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing image files.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--llama_path", type=str, required=True, help="Path to the llama model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images.")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the results JSON.")
    return parser.parse_args()

def main(args):
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(gpu_id)}")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(args.model_name, use_cache=True)
    config.llama_path = args.llama_path

    state_dict = None
    if config.llama_path != args.model_name:
        weight_map_index = json.load(open(os.path.join(args.llama_path, "pytorch_model.bin.index.json"), "r"))
        shard_files = list(set(weight_map_index["weight_map"].values()))
        state_dict = {}
        for shard_file in shard_files:
            state_dict.update(torch.load(os.path.join(args.llama_path, shard_file), map_location="cpu"))
        peft_parameters = torch.load(os.path.join(args.model_name, "saved_parameters.pth"), map_location="cpu")
        for k, v in peft_parameters.items():
            state_dict[k] = v

    model = AutoModelForCausalLM.from_pretrained(None, config=config, state_dict=state_dict)
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, padding_side='left')

    image_processor = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    results = infer_process(args.json_folder, args.image_folder, args.output_dir, model, tokenizer, image_processor)

    with open(args.output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    print(f"結果已儲存到 {args.output_file}")

def infer_process(json_folder, image_folder, output_dir, model, tokenizer, image_processor):
    image_scores = []
    for i in (1,101): 
        print(f"Processing image set {i}")
        json_path = os.path.join(json_folder, f"{i}.json")
        image_path_name = os.path.join(image_folder, f"{i}.json")
        for j in range(8):
            image_path = os.path.join(image_path_name, "gc7.5-seed0-alpha0.8", f"{j}_xl_s0.4_n20.png")
            if not os.path.exists(json_path) or not os.path.exists(image_path):
                print(f"File missing for index {i}: {json_path} or {image_path}")
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            captions = [anno["caption"] for anno in data["annos"]]
            if not captions:
                print(f"No captions found in {json_path}")
                continue

            question_template = "Please answer how many objects are in the image. For example, if there are two objects, answer 2."
            question = f"{question_template} The image contains the following objects: {', '.join(captions)}."

            output = infer(image_path, question, model, tokenizer, image_processor, output_dir)
            print(f"Output: {output}")
            try:
                score = float(output) / 3
                result = {
                    f"Image_{i}_{j+1}": image_path,
                    "Captions": captions,
                    "Prompt": question,
                    "Output": output,
                    "Score": score
                }
                image_scores.append(result)
            except ValueError:
                print(f"Invalid output for image {i}_{j}: {output}")
                continue

    return image_scores

def infer(image_path, question, model, tokenizer, image_processor, output_dir):
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)
