"""
This script creates a CLI demo with transformers backend for the glm-4-9b model,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

Note: The script includes a modification to handle markdown to plain text conversion,
ensuring that the CLI interface displays formatted text correctly.

If you use flash attention, you should install the flash-attn and  add attn_implementation="flash_attention_2" in model loading.
"""

import os
import torch
from threading import Thread
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel

# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')

## If use peft model.
# def load_model_and_tokenizer(model_dir, trust_remote_code: bool = True):
#     if (model_dir / 'adapter_config.json').exists():
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model.peft_config['default'].base_model_name_or_path
#     else:
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model_dir
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
#     )
#     return model, tokenizer


# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)

# model = AutoModel.from_pretrained(
#     MODEL_PATH,
#     trust_remote_code=True,
#     load_in_4bit=True,
#     # quantization_config = int,
#     # attn_implementation="flash_attention_2", # Use Flash Attention
#     # torch_dtype=torch.bfloat16, #using flash-attn must use bfloat16 or float16
#     # torch_dtype=torch.int8,
#     device_map="auto").eval()


class StopOnTokens(StoppingCriteria):

    def __init__(self, model):
        self.model = model
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = self.model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class GLMChatbot:
    def __init__(self, model_path=None, max_length=8192, top_p=0.8, temperature=0.6):
        self.model_path = model_path or os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')
        self.max_length = max_length
        self.top_p = top_p
        self.temperature = temperature
        self.history = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map="auto"
        ).eval()
        
        self.stop = StopOnTokens(model=self.model)

    def generate_response(self, user_input):
        self.history.append([user_input, ""])

        # Prepare messages for the chat history
        messages = []
        for idx, (user_msg, model_msg) in enumerate(self.history):
            if idx == len(self.history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

        # Prepare model input
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Stream the response
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            timeout=80,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generate_kwargs = {
            "input_ids": model_inputs,
            "streamer": streamer,
            "max_new_tokens": self.max_length,
            "do_sample": True,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "stopping_criteria": StoppingCriteriaList([self.stop]),
            "repetition_penalty": 1.2,
            "eos_token_id": self.model.config.eos_token_id,
        }

        return self.print_respond(streamer, generate_kwargs)

    def print_respond(self,streamer, generate_kwargs):
        # Start generation in a thread and capture the response
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        response = ""
        # print("GLM-4:", end="", flush=True)
        for new_token in streamer:
            if new_token:
                response += new_token
                # print(new_token, end="", flush=True) ## Fix it later
                self.history[-1][1] += new_token

        self.history[-1][1] = self.history[-1][1].strip()
        return response.strip()

    def clear_history(self):
        self.history = []





if __name__ == "__main__":
    chatbot = GLMChatbot()

    # Begin interaction
    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot.generate_response(user_input)
        # print("GLM-4:", response)
