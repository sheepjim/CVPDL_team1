import argparse
import os
import sys
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.PixelLM import PixelLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from model.segment_anything.utils.transforms import ResizeLongestSide
import json
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from transformers import TextStreamer

import numpy as np

def calculate_iou(pred_mask, bbox_mask):
    # Calculate intersection: where both masks are True
    intersection = np.logical_and(pred_mask, bbox_mask).sum()

    # Calculate union: where either of the masks is True
    union = np.logical_or(pred_mask, bbox_mask).sum()
    
#    print("bbox_mask: ", bbox_mask.sum())
#    print("pred_mask: ", pred_mask.sum())
    print("union: ", union)
    print("intersection: ", intersection)

    # Avoid division by zero if there is no union
    if union == 0:
        return 0.0

    # Calculate IoU
    iou = intersection / union
    return iou

def generate_bbox_mask(bbox, image_size=(512, 512)):
    # Unpack the image size
    image_height, image_width = image_size

    # Extract bbox parameters
    xmin, ymin, width, height = bbox

    # Convert to pixel coordinates
    xmax = xmin + width
    ymax = ymin + height
    # Ensure bounds are within image dimensions
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image_width, xmax)
    ymax = min(image_height, ymax)
    print(xmin, ymin, xmax, ymax)
    # Create the mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 1  # Set the bbox region to 1
    return mask

def generate_bbox_mask2(bbox, image_size=(512, 512)):
    # Unpack the image size
    image_height, image_width = image_size

    # Extract bbox parameters
    x_center, y_center, width, height = bbox

    # Convert to pixel coordinates
    xmin = int((x_center - width / 2) )
    ymin = int((y_center - height / 2))
    xmax = int((x_center + width / 2) )
    ymax = int((y_center + height / 2) )
    # Ensure bounds are within image dimensions
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image_width, xmax)
    ymax = min(image_height, ymax)
    print(xmin, ymin, xmax, ymax)
    # Create the mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 1  # Set the bbox region to 1
    return mask

def prompt_apply(_input):
    template_1 = """
Below is a situation, together with an object. Please help me segement them out.

Situation: {},

Objects: 1.{}.
"""
    template_2 = """
Below is a situation, together with two objects. Please help me segement them out.

Situation: {},

Objects: 1.{}, 2.{}.
"""
    template_3 = """
Below is a situation, together with three objects. Please help me segement them out.

Situation: {},

Objects: 1.{}, 2.{}, 3{}.
"""

    template_4 = """
Below is a situation, together with four objects. Please help me segement them out.

Situation: {},

Objects: 1.{}, 2.{}, 3{}, 4{}.
"""
    length = len(_input["annos"])
    if length == 1:
        output = template_1.format(_input["caption"], _input["annos"][0]["caption"])
    elif length == 2:
        output = template_2.format(_input["caption"], _input["annos"][0]["caption"],  _input["annos"][1]["caption"])
    elif length == 3:
        output = template_3.format(_input["caption"], _input["annos"][0]["caption"],  _input["annos"][1]["caption"],  _input["annos"][2]["caption"])
    else:
        output = template_4.format(_input["caption"], _input["annos"][0]["caption"],  _input["annos"][1]["caption"],  _input["annos"][2]["caption"], _input["annos"][3]["caption"])
    return output

bbox_list = []
def extract_sample(prompt_path, image_path):
    prompts = []
    image_paths = []
    for i in range(1,2):
        with open(os.path.join(prompt_path, f"{i}.json")) as f:
            data = json.load(f)
            prompt = prompt_apply(data)
            for i in range(8):
                prompts.append(prompt)
                bbox_list.append(data["annos"])
    # image_path/10.json/gc7.5-seed0-alpha0.8/0_xl_s0.4_n20.png
    for i in range(1,2):
        for j in range(8):
            _image_path = os.path.join(image_path, f"{i}.json", "gc7.5-seed0-alpha0.8", f"{j}_xl_s0.4_n20.png")
            image_paths.append(_image_path)
    return prompts, image_paths




def parse_args(args):
    parser = argparse.ArgumentParser(description="PixelLM chat")
    parser.add_argument("--version", default="xinlai/PixelLM-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--prompt_path", default="prompts_path", type=str)
    parser.add_argument("--image_path", default="image_path", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)
    parser.add_argument("--preprocessor_config", default='', type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=224, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--pad_train_clip_images", action="store_true", default=False)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)

    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if args.seg_token_num*args.image_feature_scale_num == 1:
        num_added_tokens = tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num*args.image_feature_scale_num)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    # import pdb;pdb.set_trace()
    kwargs = {"torch_dtype": torch_dtype,  "seg_token_num": args.seg_token_num, "image_feature_scale_num": args.image_feature_scale_num, "pad_train_clip_images": args.pad_train_clip_images,"resize_vision_tower": args.resize_vision_tower,
                "resize_vision_tower_size": args.resize_vision_tower_size,
                "vision_tower_for_mask": args.vision_tower_for_mask,
                "separate_mm_projector": args.separate_mm_projector,
                
                }
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
 
    model = PixelLMForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx,  **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower) if args.preprocessor_config == '' else CLIPImageProcessor.from_pretrained(args.preprocessor_config)
    transform = ResizeLongestSide(args.image_size)
    if args.pad_train_clip_images:
        transform_clip = ResizeLongestSide(clip_image_processor.size['shortest_edge'])
    model.eval()

    prompts, image_paths = extract_sample(args.prompt_path, args.image_path)
    results = dict()
    for idx, (prompt, image_path) in tqdm(enumerate(zip(prompts, image_paths)), total=800):
        results[f"{int(idx/8)+1}_{idx%8}"] = dict()
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        if args.pad_train_clip_images:
            image_clip = transform_clip.apply_image(image_np)
            clip_resize = image_clip.shape[:2]
            image_clip = preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), img_size=clip_image_processor.size['shortest_edge'])
            image_clip = image_clip.unsqueeze(0).cuda()
        else:
            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )
            clip_resize = image_clip.shape[-2:]
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        clip_resize = [clip_resize]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        

        output_ids, pred_masks, _, _ = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            clip_resize_list=clip_resize,
            original_size_list=original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer

        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        results[f"{int(idx/8)+1}_{idx%8}"]["text_output"] = text_output
        print("text_output: ", text_output)
        results[f"{int(idx/8)+1}_{idx%8}"]["prediction_rate"] = 0.
        results[f"{int(idx/8)+1}_{idx%8}"]["box_iou"] = 0.
        # import pdb;pdb.set_trace()
        for i, _pred_mask in enumerate(pred_masks):
            if _pred_mask.shape[0] == 0:
                continue
            for j, pred_mask in enumerate(_pred_mask):
                pred_mask = pred_mask.float().detach().cpu().numpy()
                pred_mask = pred_mask > 0
                np.save(f"mask_result/{int(idx/8)+1}_{idx%8}_{j}.npy", pred_mask)
                try:
                    bbox_mask = generate_bbox_mask(bbox_list[idx][j]["bbox"])
                except:
                    # Ignore segemented object > len(bbox_list[idx])
                    continue
                all_mask = generate_bbox_mask((0,0,512,512))
                all_iou = calculate_iou(pred_mask, all_mask)
                box_iou = calculate_iou(pred_mask, bbox_mask)
                np.set_printoptions(threshold=np.inf)

                if all_iou > 0.03:
                    results[f"{int(idx/8)+1}_{idx%8}"]["prediction_rate"] += 1.
                results[f"{int(idx/8)+1}_{idx%8}"]["box_iou"] += box_iou

                save_path = "{}/{}_{}_{}_mask_{}.jpg".format(
                    args.vis_save_path, image_path.split("/")[-1].split(".")[0], int(idx/8)+1, idx%8, j
                )
                cv2.imwrite(save_path, pred_mask * 100)

                save_path = "{}/{}_{}_{}_gt_mask_{}.jpg".format(
                    args.vis_save_path, image_path.split("/")[-1].split(".")[0], int(idx/8)+1, idx%8, j
                )
                cv2.imwrite(save_path, bbox_mask * 100)
                print("{} has been saved.".format(save_path))

                save_path = "{}/{}_{}_{}_masked_img_{}.jpg".format(
                    args.vis_save_path, image_path.split("/")[-1].split(".")[0], int(idx/8)+1, idx%8, j
                )
                save_img = image_np.copy()
                save_img[pred_mask] = (
                    image_np * 0.5
                    + pred_mask[:, :, None].astype(np.uint8) * np.array([0, 0, 255]) * 0.5
                )[pred_mask]
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, save_img)
                print("{} has been saved.".format(save_path))
        results[f"{int(idx/8)+1}_{idx%8}"]["prediction_rate"] /= len(bbox_list[idx])
        results[f"{int(idx/8)+1}_{idx%8}"]["box_iou"] /= len(bbox_list[idx])

        with open("result.json", 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1:])
