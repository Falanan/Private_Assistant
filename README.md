# False_Vtuber

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

* Instructor ([Ken Pu](https://kenpu.ca/))
* Course website ([Machine Learning 2: Advanced Topics](https://csci4052u.science.ontariotechu.ca/))

## Introduction

In this project, I'm going to build a false vtuver. This project will use comments made during a live streaming as input to a language model, which gives a textual response. After that the replies will be feed into a TTS(Text-to-Speech) model, this TTS model can imitate someone's voice, just like a mocking bird. This modle can read the text. The next step is to mimic the changes in the shape of a human's mouth while speaking. It can also works as a private voice assistant too!

## Technologies

* NLP Technologies:
  1. ([ChatGLM](https://github.com/THUDM/ChatGLM-6B/blob/main/README_en.md)) Chat LLM
  2. ([GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS?tab=readme-ov-file)) Text-to-Speech Model, this model reacts like a mocking bird, it can imitate someone's voice.
* Computer Vision Technologies:
  1. ([Wav2Lip](https://github.com/Rudrabha/Wav2Lip)) Speech-to-Lip Movement
  2. ([MuseTalk](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file)) Alternative for Wav2Lip
  3. Stable Diffusion: Generate Vtuber image

## Tested environment

* TODO

## Data collection

* LLM: Pretrained model from ChatGLM project.
* Text-to-Speech: Collect game charactor audios or voice actors posts.
* Wav2Lip: Waiting permission to ([LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html))dataset. If no respond, I'll use open source MuseTalk pretrained model
* Stable Diffusion: Pretrained model and LORAs to generate the chatactor image for this project.

## Model Training / Model collection

### GPT-SoVITS

Training SoVITS model is pretty simple and stright forward this project provided a perfect webui interface that can help we get all data prepared and training the model without coding, just simply clickings!!! As long as we cloned the project correctly we can run following command in terminal: `python webui.py`  Then it will bring us into a website. We can do all our operations there.

**Data preperation (Fatch Dataset tag):**

1. Vocal separation, DeEcho, DeReverberation.
2. Slice the source audio into small pieces. No attribute on the webui need to be changed, this will slice the audio into 10s pieces for better training propose.
3. Denoise all audio files.
4. Speech recognition. Generate transcript for all audio files.
5. Correct all transcripts. Machine may make mistakes.

**Fine Tuning (GPT-SoVITS-TTS tag):**

1. Dataset formating.
2. Fine-tuned Training. In this step, we will train both SoVITS model and GPT model.

Based on personal interests, I collected several celebrities voices, models will be availabe soon on Hugging Face.

**Please be awared of that I have no control of further usage of open source models**

Fine tuned models list:

* [X]  Zhang Yaqian(张雅倩): ([抖音](https://v.douyin.com/ikvoFcd7/)). 6 hours streaming recording.
* [X]  chengzimiaoj(橙子喵酱)：([Social Media](https://linktr.ee/chengzimiaoj)). 40 minutes streaming recording.
* [ ]  Linvo takls about cosmos(Linvo说宇宙): ([Bilibili](https://space.bilibili.com/357515451?spm_id_from=333.337.0.0)).

### Wav2Lip
