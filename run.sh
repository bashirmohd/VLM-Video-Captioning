VIDEO_DIR_PATH=videos
OUTPUT_DIR_PATH=predictions
CKPT_PATH=all_checkpoints

# Parameters
run_in_env() {
    local env="$1"
    shift
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$env"
    "$@"
}

# Save video key frames
run_in_env vllm
    python save_key_frames.py --video_input_dir_path "$VIDEO_DIR_PATH" --frames_dir_path "$OUTPUT_DIR_PATH/keyframes"

# Caption all the extracted keyframes
run_in_env llava
    python image_captioning.py --key_frame_dir_path "$OUTPUT_DIR_PATH/keyframes" --output_dir_path "$OUTPUT_DIR_PATH" --llava_model_path "$CKPT_PATH/llava-v1.5-7b"

# Detect objects in all the keyframes
run_in_env eva
    python object_detection_eva.py --key_frame_dir_path "$OUTPUT_DIR_PATH/keyframes" --output_dir_path "$OUTPUT_DIR_PATH"

# Use LLM to generate coherent video caption
run_in_env vllm
    python llm_video_captioning.py --predictions_dir_path "$OUTPUT_DIR_PATH" --output_dir_path "$OUTPUT_DIR_PATH" --model_path "$CKPT_PATH/vicuna-13b-v1.5"
