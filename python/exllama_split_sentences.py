
#This file iterates over full narratives and 
#1) codes each chunk as a fragment, sentence, or sentences
#2) tries to split the chunk no matter what into a numbered list of sentences
#You then go back to the downloads folder and clean and create a final sentence list.

import numpy as np
generator=icbe_llm_generator()

#pip install pyread
import pyreadr
crisis_narratives = pyreadr.read_r("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/crises_narratives_rex_2023_webscrape.Rds").popitem()[1]
print(crisis_narratives.keys())


#73 one of the longest at 15152 characters

#Alright new plan, we're going to seperate on newline characters first and then try to split invidiual chunks.

for j in range(crisis_narratives.shape[0]):
  story=crisis_narratives.text[j]
  import re
  story_partial_split=re.split("[\r\n]+",story)
  story_partial_split=[q.strip() for q in story_partial_split if len(q.strip())>0]

  #Try classifying each chunk first
  def assemble_prompt(story):
    prompt1 = """###: System: You are computer program that does exactly as instructed and nothing else.\n### User: ### Begin Text\n%s\n### End Text\n### Begin Question\nWhich of these answers best describes the above text? ### End Question\n### Begin Answer Choices\n1. A sentence fragment, like a section heading.\n2. One single complete sentence.\n3. More than one complete sentence.\n### Final Answer (a single number followed by a @ symbol)\n Assistant:""" % (story)
    return(prompt1)
  
  import re
  chunk_classification=list()
  for i, chunk in enumerate(story_partial_split):
    #print(i, flush=True)
    print(chunk, flush=True)
    thischunk=chunk
    prompt1=None
    output1=None
    output_answer_review=None
    prompt1=assemble_prompt(thischunk) #edited by hand to start on a working sentence. Let's see if it hallucinates.
    #Oh wow. WHen you pass it a broken first sentence it just halucinates a totally different story. Lol
    output1 =  generator.generate_simple_rex(prompt1, max_new_tokens = np.floor((len(chunk)/4) + 100 ).astype('int')  , custom_stop= '@'  ) #
    output_answer_review = output1.replace(prompt1, "")
    output_answer_review=output_answer_review.strip()
    print(output_answer_review, flush=True)
    chunk_classification.append(output_answer_review)
    
  
  def assemble_prompt(story):
    prompt1 = """###: System: You are computer program that does exactly as instructed and nothing else.\n### User: ### Begin Text\n%s\n### End Text\n### Begin Question\nSplit the above text into a numbered list of individual sentences.### End Question\n### Final Answer (a numbered list of split sentences followed by a @ symbol)\n Assistant:""" % (story)
    return(prompt1)
  
  sentence_splits=list()
  for i, chunk in enumerate(story_partial_split):
    #print(i, flush=True)
    print(chunk, flush=True)
    thischunk=chunk
    prompt1=None
    output1=None
    output_answer_review=None
    prompt1=assemble_prompt(thischunk) #edited by hand to start on a working sentence. Let's see if it hallucinates.
    #Oh wow. WHen you pass it a broken first sentence it just halucinates a totally different story. Lol
    output1 =  generator.generate_simple_rex(prompt1, max_new_tokens = np.floor((len(chunk)/4) + 100 ).astype('int')  , custom_stop= '@'  ) #
    output_answer_review = output1.replace(prompt1, "")
    output_answer_review=output_answer_review.strip()
    print(output_answer_review, flush=True)
    sentence_splits.append(output_answer_review)
  
  import pandas as pd
  df = pd.DataFrame({'chunks':story_partial_split, 'chunks_class':chunk_classification, 'sentences_raw':sentence_splits})
      
  df.to_csv("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_out/crises_split/crisis_"+str(j)+".csv")




    

