# Automatic Video-Caption Generation

# Create conda environments
1. LLaVA (For key frame captioning)

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout v1.1.3
pip install -e .

git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
```


# EVA (For object detection in key frames)

Please follow instructions at [https://github.com/baaivision/EVA/tree/master/EVA-02/det#setup](https://github.com/baaivision/EVA/tree/master/EVA-02/det#setup).


# VLLM (Environment setup for LLM inference)
```bash
conda create --name vllm python=3.10

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install "fschat[model_worker,webui]"
pip install vllm
```