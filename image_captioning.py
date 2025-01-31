import json
import os
import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import numpy as np
import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
from tqdm import tqdm
import multiprocessing
from random import shuffle
from multiprocessing.pool import Pool
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--key_frame_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)

    parser.add_argument("--llava_model_path", required=True)

    args = parser.parse_args()

    return args


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_prompt(model, qs, conv_mode="llava_v1", ):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    return prompt, stop_str


def get_caption_prompt(model, conv_mode="llava_v1"):
    qs = "Generate a single sentence caption of the provided image."

    return get_prompt(model, qs, conv_mode)


def run_inference(key_frame_dir_path, output_dir_path, llava_model_path):
    # Create lava model
    model_name = get_model_name_from_path(llava_model_path)
    model = load_pretrained_model(llava_model_path, None, model_name)

    tokenizer, model, image_processor, context_len = model
    for video_keyframe_dir in tqdm(os.listdir(key_frame_dir_path)):
        video_tags_dict = {}
        video_name = video_keyframe_dir
        video_tags_dict[video_name] = {}
        video_keyframe_dir_path = os.path.join(key_frame_dir_path, video_keyframe_dir)
        all_data = {}
        for i, frame in enumerate(os.listdir(video_keyframe_dir_path)):
            all_data[frame] = {}
            image_path = f"{video_keyframe_dir_path}/{frame}"
            # Load image
            image = load_image(image_path)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

            # Prepare the prompt for main landmark
            cation_prompt, stop_str = get_caption_prompt(model)
            input_ids = tokenizer_image_token(cation_prompt, tokenizer,
                                              IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # Run LLaVA inference
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            # Post process
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs_caption = outputs.strip().lower()

            all_data[frame] = outputs_caption

        with open(f"{output_dir_path}/llava/{video_name}.json", 'w') as f:
            json.dump(all_data, f)


def main():
    args = parse_args()
    key_frame_dir_path = args.key_frame_dir_path
    output_dir_path = args.output_dir_path
    llava_model_path = args.llava_model_path

    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(f"{output_dir_path}/llava", exist_ok=True)

    run_inference(key_frame_dir_path, output_dir_path, llava_model_path)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
