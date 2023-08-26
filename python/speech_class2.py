
import numpy as np
generator=icbe_llm_generator()

import pandas as pd
icbe_llm_codebook = pd.read_csv("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/icbe_llm_codebook.csv" , na_filter=False) #make sure you set na to ''

df_codebook = icbe_llm_codebook[icbe_llm_codebook.Variable=='Speech Class'].drop('ICBE1Code', axis=1).reset_index(drop=True) #need to drop index

n=df_codebook.shape[0]
df_codebook.columns
df_dictionary={ 'Variable': df_codebook['Variable'][0].strip(),
  'Description': df_codebook['Description'][0].strip(),
  'Usage Notes': df_codebook['Usage Notes'][0].strip(),
  'Response Options':
    [{'Response':df_codebook['Response Option'][i].strip(),'Response Description':df_codebook['Response Description'][i].strip(),'Positive Examples':df_codebook['Positive Examples'][i].strip().split("\n")} for i in range(0,n)] #,'Negative Examples':df_codebook['Negative Examples'][i]
}
import pprint
pp = pprint.PrettyPrinter(indent=1)
pp.pprint(df_dictionary)

import json
df_json =json.dumps(df_dictionary)
#question=df_json

question = df_codebook['Description'][0].strip() +"\n" + "\n".join([df_codebook['response_number'][i].strip() + ") " +  df_codebook['Response Option'][i].strip() +" - " + df_codebook['Response Description'][i].strip()  for i in range(0,n)])

def assemble_prompt(story, sentence, question, answer_review=None, debate=None, output_final_answer=None):
  
  prompt1 = """USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question\n%s\n### End Question\n  Carefully reread the specific sentence and the question. Write a short paragaph debating the merrits of each response, refering specifically to the coding instructions and how they determine the answer in this case, and conluding with a synthesis in favor of only one answer. Be factually correct and very concise. Think carefully about the entire question.\n""" % (story, sentence, question)  #### Begin Answer (single answer) 
  prompt2 = prompt1 + """%s\n### End List of Available Answers\n### Begin Debate (Carefully reread the specific sentence and the question. Write a short paragaph debating the merrits of each response, refering specifically to the coding instructions and how they determine the answer in this case, and conluding with a synthesis in favor of only one answer. Be factually correct and very concise. Think carefully about the entire question.)\n""" % (answer_review)
  #prompt3 = prompt2 + """"%s\n### End Debate\n### Machine Readable Final Answer (Write the final answer choice on a new line by itself. Terminate immediately.)\n""" % (debate)
  #prompt3 = prompt2 + """%s\n### End Debate\n### Final Answer (Reread the Specific Sentence, the coding instructions of the question, and the debate and synthesize a single final answer.)\n"""  % (debate)
  #prompt4 = prompt3 + """%s\n### End Final Answer\n### Machine Readable Final Answer (Write the final answer choice on a new line by itself. Terminate immediately.)\n""" % (output_final_answer)

  if answer_review==None and debate==None and output_final_answer==None: 
    return(prompt1 + """ASSISTANT: """) #Always tack on assistant right at the end
  #if answer_review!=None and debate==None and output_final_answer==None:
  #  return(prompt2 + """ASSISTANT: """)
  #if answer_review!=None and debate!=None and output_final_answer==None:
  #  return(prompt3 + """ASSISTANT: """)
  #if answer_review!=None and debate!=None and output_final_answer!=None:
  #  return(prompt4 + """ASSISTANT: """)
# 

#assemble_prompt(story="practice story", sentence="practice sentence", question="practice question", answer_review="practice answer review", debate="practice debate", output_final_answer="practice final answer")


#pip install pyread
import pyreadr
import pandas as pd
crisis_narratives = pyreadr.read_r("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/crises_narratives_rex_2023_webscrape.Rds").popitem()[1].reset_index(drop=True)
print(crisis_narratives.keys())
#Note we've switched to events now
crisis_events = pyreadr.read_r("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/crises_narratives_rex_2023_webscrape_parsed_sentences_parsed_events.Rds").popitem()[1].reset_index(drop=True)

#If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
for crisisnum in set(crisis_events.crisno.values):
  fileout="/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_out/speech_class/crisis_"+str(crisisnum)+".csv"
  if os.path.isfile(fileout):
    continue
  print("Starting crisis " +str(crisisnum))
  story=crisis_narratives.text[int(crisisnum)-1] #remember to subtract off one
  story=story.split('References')[0].strip() #Split off references to save tokens
  crisis_events_subset = crisis_events[crisis_events.crisno==crisisnum].copy().reset_index(drop=True)
  crisis_events_subset['event_type_thought']=''
  crisis_events_subset['event_type_thought_question']=''
  crisis_events_subset['event_type_thought_answer_review']=''
  crisis_events_subset['event_type_thought_debate']=''
  crisis_events_subset['event_type_output_thought_final_answer']=''

  event_texts=crisis_events_subset.event_text.values #note we switch to event text now
  for i, event_text in enumerate(event_texts):
    sentence=event_text #we call the event text a sentence
    if crisis_events_subset['chunks_class'][i]=="fragment":
      continue
    if crisis_events_subset['event_count'][i] not in ['1 Codable Event','2 Codable Events','3 Codable Events']:
      continue
    max_tokens=2750
    print(i, flush=True)
    print(event_text, flush=True)
    crisis_events_subset.loc[i,'speech_class_question']=question
    #Prompt1
    prompt1=assemble_prompt(story, sentence, question, answer_review=None, debate=None) #trimming references off the story and neg examples from the question got us under 2700 but there are longer stories
    tostrip= round(min(max_tokens - 100 - generator.tokenizer.encode(prompt1).shape[1],0)*4.1)
    if tostrip:
      prompt1=assemble_prompt(story[:tostrip], sentence, question, answer_review=None, debate=None)
    output1 =  generator.generate_simple_rex(prompt1, max_new_tokens = 100, custom_stop= "#") #had to up for longer answers
    output_answer_review = output1.replace(prompt1, "") #.split("\n\n")[0].strip()
    crisis_events_subset.loc[i,'speech_class']=output_answer_review
    print(output_answer_review, flush=True)

  #crisis_sentences_subset['event_type_prompt_for_saving']=prompt_for_saving
  crisis_events_subset.to_csv(fileout, index=False)
