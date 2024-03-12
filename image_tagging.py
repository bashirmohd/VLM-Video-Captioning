import argparse
import torch
import os
import json
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Image tagging")

    parser.add_argument("--key_frame_dir_path", required=True, help="")
    parser.add_argument("--ckpt_path", required=False, default="recognize-anything-plus-model/ram_plus_swin_large_14m.pth", help="")
    parser.add_argument('--image-size', default=384, type=int, metavar='N',
                        help='input image size (default: 448)')
    parser.add_argument("--output_dir_path", required=True, help="")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir_path, exist_ok=True)
    os.makedirs(f"{args.output_dir_path}/tags", exist_ok=True)

    # Select the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image transformations
    transform = get_transform(image_size=args.image_size)

    # Load model
    model = ram_plus(pretrained=args.ckpt_path, image_size=args.image_size, vit='swin_l')
    model.eval()
    model = model.to(device)

    for video_keyframe_dir in tqdm(os.listdir(args.key_frame_dir_path)):
        video_tags_dict = {}
        video_name = video_keyframe_dir
        video_tags_dict[video_name] = {}
        video_keyframe_dir_path = os.path.join(args.key_frame_dir_path, video_keyframe_dir)
        for i, frame in enumerate(os.listdir(video_keyframe_dir_path)):
            image_path = f"{video_keyframe_dir_path}/{frame}"
            image = transform(Image.open(image_path)).unsqueeze(0).to(device)
            res = inference(image, model)
            video_tags_dict[video_name][i] = res[0]
        with open(f"{args.output_dir_path}/tags/{video_name}.json", 'w') as f:
            json.dump(video_tags_dict, f)
