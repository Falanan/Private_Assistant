# False_Vtuber

# Introduction

In this project, I'm going to build a fake vtuver. This project will use comments made during a live streaming as input to a language model, which gives a textual response. After that the replies will be feed into a TTS(Text-to-Speech) model, this TTS model can imitate someone's voice, just like a mocking bird. This modle can read the text. The next step is to mimic the changes in the shape of a human's mouth while speaking. It can also works as a private voice assistant.

## Technologies

* NLP Technologies:
  1. ([ChatGLM](https://github.com/THUDM/ChatGLM-6B/blob/main/README_en.md)) Chat LLM
  2. ([GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS?tab=readme-ov-file)) Text-to-Speech Model, this model reacts like a mocking bird, it can He can imitate someone's voice.
* Computer Vision Technologies:
  1. ([Wav2Lip](https://github.com/Rudrabha/Wav2Lip)) Speech-to-Lip Movement
  2. ([MuseTalk](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file)) Alternative of Wav2Lip
  3. Stable Diffusion: Generate Vtuber image

## Data collection

* LLM: Pretrained model.
* Text-to-Speech: Collect game charactor audios or voice actors posts.
* Wav2Lip: Waiting permission to ([LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html))dataset. If no respond, I'll use open source MuseTalk pretrained model
* Stable Diffusion: Pretrained model and LORAs to generate the chatactor image for this project.
