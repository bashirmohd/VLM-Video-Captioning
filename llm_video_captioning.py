import argparse
import os
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from fastchat.model import get_conversation_template


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--predictions_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--model_path", required=False, default="vicuna-7b-v1.5")

    args = parser.parse_args()

    return args


def get_scene_graph(video_captions, video_detections):
    # Sort the frames to process them in chronological order
    sorted_frames = sorted(video_captions.keys())

    scene_graph = "Video Scene Graph:\n"

    for i, frame in enumerate(sorted_frames):
        # Retrieve the caption and detections for the current frame
        caption = video_captions.get(frame, "No caption")
        detections = video_detections.get(frame, [])

        # Construct the scene description for the current frame
        scene_description = f"Frame {i}: {caption}\n"
        scene_description += f"Number of objects: {len(detections)}\n"

        # If there are detections, list their labels
        if detections:
            labels = [detection['label'] for detection in detections]
            scene_description += f"Objects: {', '.join(labels)}\n"
        else:
            scene_description += "Objects: None\n"

        # Add the current frame's description to the scene graph
        scene_graph += scene_description + "\n"

    return scene_graph


def get_vicuna_chat_prompt(scene_graph):
    prompt = f"""
Given the detailed scene graph below, which outlines key events and objects detected in a sequence of video frames, generate a coherent and engaging video caption. The caption should summarize the main activities and notable changes throughout the video. Highlight significant moments or transitions, such as changes in the number of objects, or objects that appear or disappear as the video progresses. The goal is to create a caption that provides a clear summary of the video's content, enabling an understanding of the key events and dynamics without needing to watch the video itself.

Scene Graph:
{scene_graph}

Please generate a coherent video caption that encapsulates the essence of the video, paying particular attention to significant changes and moments.
"""
    return prompt


def get_chat_completion_prompt(model_path, scene_graph):
    prompt = get_vicuna_chat_prompt(scene_graph)
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def run_vicuna_inferene(model_path, model, sampling_params, scene_graphs):
    prompts = [get_chat_completion_prompt(model_path, scene_graph) for scene_graph in scene_graphs]

    outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    responses, reasons = [], []
    for output in outputs:
        responses.append(output.outputs[0].text)
        reasons.append(output.outputs[0].finish_reason)

    return responses, reasons


def main():
    args = parse_args()
    # Create output directory path if not exists
    output_path = f"{args.output_dir_path}/video_captions"
    os.makedirs(output_path, exist_ok=True)

    # Create sampling params & load Vicuna Model
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
    model = LLM(model=args.model_path, max_num_batched_tokens=4096)

    captions_dir_path = f"{args.predictions_dir_path}/llava"
    detections_dir_path = f"{args.predictions_dir_path}/eva-02"

    all_videos = os.listdir(captions_dir_path)

    for video in tqdm(all_videos):
        video_dense_captions = {}
        video_name = video[:-5]

        video_captions_path = f"{captions_dir_path}/{video}"
        video_detections_path = f"{detections_dir_path}/{video}"
        video_captions = json.load(open(video_captions_path, 'r'))
        video_detections = json.load(open(video_detections_path, 'r'))

        scene_graph = get_scene_graph(video_captions, video_detections)
        print(scene_graph)

        dense_captions, vllm_stop_reasons = run_vicuna_inferene(args.model_path, model, sampling_params,
                                                                [scene_graph])

        video_dense_captions[video_name] = dense_captions[0]

        with open(f"{output_path}/{video}", 'w') as f:
            json.dump(video_dense_captions, f)


if __name__ == "__main__":
    main()
