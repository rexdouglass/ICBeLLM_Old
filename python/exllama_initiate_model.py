#https://github.com/turboderp/exllama/blob/master/example_basic.py
#!pip install flash-attn --no-build-isolation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import os
os.getcwd()
import sys
sys.path.insert(0, "/home/skynet3/Downloads/exllama/")
from model import ExLlama, ExLlamaCache, ExLlamaConfig

from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
import torch

def icbe_llm_generator():

  #You are a natural language processing pipeline. You extract entities from text. Parse the text carefully and return every single person, place, and thing mentioned in the text.
  
  #I'm processing prompts at about 41 tokens a second and producing responses at about 14 tokens a second
  #/home/skynet3/Downloads/exllama
  #python test_benchmark_inference.py -d <path_to_model_files> -p -ppl
  #python example_chatbot.py -d "/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_temp/Wizard-Vicuna-30B-Uncensored-GPTQ/" -un "Jeff" -p prompt_chatbort.txt
  #python webui/app.py -d "/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_temp/Wizard-Vicuna-30B-Uncensored-GPTQ/"
  
  #python webui/app.py -d "/home/skynet3/Downloads/LLAMA/StableBeluga2-GPTQ/" -gs 17.2,24
  #python webui/app.py -d "/home/skynet3/Downloads/LLAMA/airoboros-l2-70B-gpt4-1.4.1-GPTQ/" -gs 17.2,24 -length 4096
  
  #model_directory =  "/home/skynet3/Downloads/LLAMA/airoboros-l2-70B-gpt4-1.4.1-GPTQ/"
  model_directory =  "/home/skynet3/Downloads/LLAMA/StableBeluga2-GPTQ-4ibt-32g/"
  model_directory =  "/home/skynet3/Downloads/LLAMA/StableBeluga2-GPTQ-4ibt-32g/"
  
  # Directory containing model, tokenizer, generator
  
  #model_directory =  "/mnt/str/models/llama-13b-4bit-128g/"
  #model_directory =  "/home/skynet3/Downloads/LLAMA/Llama-2-13B-chat-GPTQ/"
  #model_directory =  "/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_temp/Wizard-Vicuna-30B-Uncensored-GPTQ/"
  #model_directory =  "/home/skynet3/Downloads/LLAMA/LLaMA-30b-GPTQ/"
  #model_directory =  "/home/skynet3/Downloads/LLAMA/guanaco-33B-GPTQ/"
  #model_directory =  "/home/skynet3/Downloads/LLAMA/falcon-40b-instruct-3bit-GPTQ/"
  
  # Locate files we need within that directory
  
  tokenizer_path = os.path.join(model_directory, "tokenizer.model")
  model_config_path = os.path.join(model_directory, "config.json")
  st_pattern = os.path.join(model_directory, "*.safetensors")
  model_path = glob.glob(st_pattern)[0]
  
  # Create config, model, tokenizer and generator
  
  config = ExLlamaConfig(model_config_path)               # create config from config.json
  config.model_path = model_path                          # supply path to model weights file
  
  #Rex's special additions
  config.set_auto_map("17.2,24") #This did make it allocate to both #https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/21
  #config.set_auto_map("16.2,24") #"15.5. 24 is what I use.""  https://github.com/turboderp/exllama/issues/191
  #config.max_input_len = 4096 #4096 
  config.max_seq_len   = 4096 #I don't understand the difference between these two.
  config.flash_attn = 4096 #experimenting to see if this works #this one is wire to input length.
  
  model = ExLlama(config)                                 # create ExLlama instance and load the weights
  tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
  
  tokenizer.encode('#') #396
  tokenizer.eos_token
  tokenizer.eos_token_id
  tokenizer.encode(tokenizer.eos_token)
  
  cache = ExLlamaCache(model)                             # create cache for inference
  
  # monkey patch generator simple to have a custom stop token
  def generate_simple_rex(self, prompt, max_new_tokens = 128, custom_stop=None):
      self.end_beam_search()
      ids, mask = self.tokenizer.encode(prompt, return_mask = True)
      self.gen_begin(ids, mask = mask)
      max_new_tokens = min(max_new_tokens, self.model.config.max_seq_len - ids.shape[1])
      eos = torch.zeros((ids.shape[0],), dtype = torch.bool)
      for i in range(max_new_tokens):
        token =  generator.gen_single_token(mask = mask)
        token_as_string =  generator.tokenizer.decode( token )[0]
        #print(token_as_string)
        if custom_stop in token_as_string:
          #print("breaking!")
          generator.sequence=generator.sequence[0,:-1] #strip off that last token
          break
        for j in range(token.shape[0]):
          if token[j, 0].item() ==  generator.tokenizer.eos_token_id: eos[j] = True
        if eos.all(): break
      text = self.tokenizer.decode(self.sequence[0] if self.sequence.shape[0] == 1 else self.sequence)
      return text
    
  ExLlamaGenerator.generate_simple_rex = generate_simple_rex #monkey patch in our change
  generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
  
  #generator.end_beam_search()
  #ids, mask = generator.tokenizer.encode("USER: Print 5 pounds signs, e.g. '#####' ASSISTANT:", return_mask = True)
  #generator.gen_begin(ids, mask)
  #token_as_string=tokenizer.decode(generator.gen_single_token(mask = mask))
  #'#####' #is a token. Hilarious.
  
  tokenizer.eos_token_id
  tokenizer.encode('a')
  tokenizer.decode(torch.tensor([[2]]))
  # Configure generator
  
  generator.disallow_tokens([tokenizer.eos_token_id])
  
  generator.settings.token_repetition_penalty_max = 1.0 #ok if you lower this to 0 it just repeats over and over again
  generator.settings.temperature = 0.01 #0.95
  generator.settings.top_p = 1.0
  generator.settings.top_k = 40
  generator.settings.typical = 0.5
  
  return(generator)




# Produce a simple generation

#prompt = "Once upon a time,"
#print (prompt, end = "")
#output = generator.generate_simple(prompt, max_new_tokens = 100)
#print(output[len(prompt):])
#[output]
#generator.generate_simple_rex("USER: Print 5 pounds signs, e.g. '#####' ASSISTANT:", max_new_tokens = 10, custom_stop="#" ) #
