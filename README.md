# Private_Assistant

## About Me

* Wenbo Zhang ([LinkedIn](https://www.linkedin.com/in/wenbo-zhang-falana/))

4th Year Computer Science Student Passion in Computer Vision | Machine Learning | AI

Everything about High Performance Computing!

**!!!!!!!!!!!  HPC MAGIC  !!!!!!!!!!**

## About this course

**CSCI 4052U Machine Learning 2: Advanced Topics**

This course is for students with a foundational understanding of machine learning. It explores advanced topics including, but not limited to, encoder/decoder architectures, attention mechanisms, and transformer-based models. The course demonstrates the application of these advanced architectures in creating state-of-the-art neural networks for various applications, such as language modeling (both masked and generative), computer vision, text-to-speech, speech recognition, multimodal learning, and Q-learning in agent-based AI systems. Additionally, it covers essential methodologies for developing and deploying AI systems, encompassing aspects like data pipelines,
model management, training and fine-tuning, quantization, model distillation, knowledge infusion,
and performance monitoring.

* Instructor ([Ken Q. Pu](https://kenpu.ca/))
* Course website ([Machine Learning 2: Advanced Topics](https://csci4052u.science.ontariotechu.ca/))

## Introduction

In this project, I'm going to build a private assistant. This model contains 3 main components: Large language models, Text-to-Speech models, Wav-to-Lip models.

## Technologies

* NLP Technologies:
  1. ~~([ChatGLM](https://github.com/THUDM/ChatGLM-6B/blob/main/README_en.md))  - A Chinese specified Large Language Model~~
  2. ([GLM-4](https://github.com/THUDM/GLM-4/blob/main/README_en.md)) - Large Language Model works for Multi - Language
  3. ([GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS?tab=readme-ov-file)) Text-to-Speech Model, this model reacts like a mocking bird, it can imitate someone's voice.
* Computer Vision Technologies:
  1. ([Wav2Lip](https://github.com/Rudrabha/Wav2Lip)) Speech-to-Lip, This model works better for English
  2. ([MuseTalk](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file)) Alternative for Wav2Lip. This model works better for Asian Languages
  3. ~~Stable Diffusion: Generate Avatar images~~ (If time pemitted)

## Tested environment

* Windows11 23H2
* Python 3.10.15 under Conda environmnet
* Nvidia 2080Ti 22GB with Hardware-Accelerated GPU Scheduling enabled. Reason: When doing inference, VRAM taken may slightly exceed 22GB.

## Data collection

* LLM: Pretrained model from ChatGLM project.
* Text-to-Speech: Collect game charactor audios or voice actors posts.
* Wav2Lip: Trained on ([LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html))dataset.
* Stable Diffusion: Pretrained model and LORAs to generate the chatactor image for this project.

## Model Training / Model collection

### GPT-SoVITS

Training SoVITS model is pretty simple and stright forward this project provided a perfect webui interface that can help we get all data prepared and training the model without coding, just simply clickings!!! As long as we cloned the project correctly we can run following command in terminal: `python webui.py`  Then it will bring us into a website. We can do all our operations there.

**Data preperation:**

1. Vocal separation, DeEcho, DeReverberation.
2. Slice the source audio into small pieces. No attribute on the webui need to be changed, this will slice the audio into 10s pieces for better training propose.
3. Denoise all audio files.
4. Speech recognition. Generate transcript for all audio files.
5. Correct all transcripts. Machine may make mistakes.

**Fine Tuning :**

1. Dataset formating.
2. Fine-tuned Training. In this step, we will train both SoVITS model and GPT model.

Based on personal interests, I collected several celebrities voices, models are availabe on [Hugging Face](https://huggingface.co/Falanan/Private_Assistant_Models).

**Please be awared of that I have no control of further usage of open source models**

Fine tuned models list:

* Chinese Sepsific

* [X]  Zhang Yaqian(张雅倩): ([抖音](https://v.douyin.com/ikvoFcd7/)), ([Red Book](https://www.xiaohongshu.com/user/profile/5ab2338b4eacab7968ac3330?xhsshare=CopyLink&appuid=61b25feb000000001000632f&apptime=1727475569)). Data source: 6 hours streaming recording.
* [X]  Orangin Neko(橙子喵酱)：([Social Media](https://linktr.ee/chengzimiaoj)). Data source: 40 minutes streaming recording.
* [ ]  Linvo takls about cosmos(Linvo说宇宙): ([Bilibili](https://space.bilibili.com/357515451?spm_id_from=333.337.0.0)).

* English Sepsific

* [X]  Genshin Impact Character: [Yae Miko](https://genshin.hoyoverse.com/en/character/inazuma?char=10). Data source: Genshin Impact game original sound track

### GLM-4

In this part of this project, I'll use GLM-4-Chat model. The original model takes around 21Gbs GPU Ram to run, but this model can run under INT4 quantize. Whis INT4 quantize the precision would be lower but only takes 8Gbs GPU Ram, which really saves space for other models.

### Wav2Lip/MuseTalk

Training a model on LRS2 dataset from scratch takes really a long time and require a really good hardwares. For this part, I'll use pretrained models first. If time permitted, I'll train a model from very beginning.

## Installation

install K-Lite Codec Pack

Note: Only PyTorch 2.1.2/2.0.1/2.4.1 works for MuseTalk required openmim libraries. I do recommend to use version 2.1.2. Since 2.4.1 keep printing out warning information that requires you to update the load model function, while open-mmlab still not update their models till Nov/01/2024. And I've put all the requirements into ```requirements.txt``` file. So you can simply run the following command in your termial to install all requirements:

``` pip install -r requirements.txt```

### mmlab packages for MuseTalk
```
pip install --no-cache-dir -U openmim 
mim install "mmengine==0.10.5"
mim install "mmcv==2.1.0" 
mim install "mmdet==3.2.0" 
mim install "mmpose==1.3.2" 
```

MuseTalk inference need to install Visual Studio, download from there: [https://visualstudio.microsoft.com/visual-cpp-build-tools](https://visualstudio.microsoft.com/visual-cpp-build-tools)

## Inference results

Note: Since Github does not support audio files(wav & mp3) embedded. So, I decided to combine two parts together.

### Wav2Lip & GPT-SoVITS
Original Sound Track Reference Text: Not bad at all. I'm glad you finally got to reveal the tricks you've been keeping up your sleeve.

Inference text: Hello master, I am Yae Miko. What can I do for you?
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="33%">Original Image</td>
        <td width="33%">Original Sound Track + Wav2Lip</td>
        <td width="33%">GPT-SoVITS + Wav2Lip</td>
  </tr>
  <tr>
    <td>
    <p>Avatar: Yae Miko</p>
      <img src=Inference_results\Originals\Yae_Miko_Avatar.png width="95%">
      <p>Credits: <a href="https://civitai.com/images/1105384">Civitai AI</a></p>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/af5f4770-744f-4f48-8ed5-6a6cb881a47b controls preload></video>
    </td>
    <td >
      <video src=https://github.com/user-attachments/assets/7cc6fd65-0f46-47c8-aa5d-31c559f78019 controls preload></video>   
    </td>
  </tr>
</table >

### MuseTalk & GPT-SoVITS
Original Text: 对啊，小猫还会外语，哎，我有个特别大的疑惑，猫和狗能听懂对方讲话吗？我感觉他们应该也有自己的语言吧。(English Translation: Yeah, kittens also speak foreign languages... hey, I have a particularly big question... can cats and dogs understand each other's speech? I feel like they should have their own language too.)

Inference Text: 主人你好，请问你需要什么帮助？(English Translation: Hello Master, What can I do for you?)

<table>
  <tr style="font-weight: bolder;text-align:center;">
        <td width="33%">Original Image</td>
        <td width="33%">Original Sound Track + MuseTalk</td>
        <td width="33%">GPT-SoVITS + MuseTalk</td>
  </tr>
  <tr>
    <td>
      <img src=Inference_results\Originals\Yaqian_original_image.jpg  width="95%">
      <p>Credits: <a href="http://xhslink.com/a/srup3KJBV3NY">Link to the post</a></p>
    </td>
    <td >
      <video src=https://github.com/user-attachments/assets/e0c7c55f-9b03-4f4f-b8ea-c0785aceaf59 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/user-attachments/assets/8d1558af-2046-4c8b-a4bc-56bd215e8c19 controls preload></video>
    </td>
  </tr>
</table >

### GLM-4
<img src=Inference_results\glm.png  width="95%">

## Thought on Reinforcement Learning
Reinforcement Learning is a method that the current output of this moment depends on previous outputs and a reward. This can be implemented on NLP part. For example, we have the records of a Language anchors. The live broadcasts consist of chatting with the chat room. We have the chat room inputs and the anchors responds. This particular person must be chatting according to their personality. So we have all the contents, and we can use these data to train a model that the model is responding you with the particular personality. 

This may can achieve by fine-tuning a pretrained model.

If time permitted, ~~I'll collect some data and try to fine tune GLM-4 model.~~


## Download Pretrained Models for different modules
### GLM-4
Auto downloaded when first time running the program. 

File path:
### GPT-SoVITS
Download the pretrained models on [Hugging face](https://huggingface.co/Falanan/Private_Assistant_Models). Then put the entire folder under Models_Pretrained folder.

### Wav2Lip
Download wav2lip_gan model provided [Wav2Lip](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) project. Put the downloaded file under Models_Pertrained/wav2lip folder.

### MuseTalk
I personally recommend follow the official link to the website. Here is the link: [MuseTalk models preparation instruction](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file#download-weights)

Here is the copy:
#### Download weights
You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MuseTalk).

2. Download the weights of other components:
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [whisper](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [face-parse-bisent](https://github.com/zllrunning/face-parsing.PyTorch)
   - [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)


Finally, these weights should be organized in `models` as follows:
```
Models_Pertrained
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```


