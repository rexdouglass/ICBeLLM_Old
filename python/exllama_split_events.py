

import numpy as np
generator=icbe_llm_generator()


#pip install pyread
import pyreadr
import pandas as pd
crisis_narratives = pyreadr.read_r("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/crises_narratives_rex_2023_webscrape.Rds").popitem()[1]
print(crisis_narratives.keys())
crisis_sentences = pyreadr.read_r("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/crises_narratives_rex_2023_webscrape_parsed_sentences.Rds").popitem()[1]

#If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
#crisisnum=196
for crisisnum in crisis_narratives.crisno.values:
  fileout="/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_out/events_split/crisis_event_split_"+str(crisisnum)+".csv"
  if os.path.isfile(fileout):
    continue
  story=crisis_narratives.text[crisisnum-1] #remember to subtract off one
  story=story.split('References')[0].strip()
  crisis_sentences_subset = pd.read_csv("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_out/number_of_events/crisis_event_count_"+str(crisisnum)+".csv")
  crisis_sentences_subset = crisis_sentences_subset[crisis_sentences_subset.crisno==crisisnum].copy().reset_index(drop=True) #this should be redundant but maybe good practice
  crisis_sentences_subset['event_split']=''
  crisis_sentences_subset['event_split_question']=''
  crisis_sentences_subset['event_split_debate']=''
  crisis_sentences_subset['event_split_output_final_answer']=''

  sentences=crisis_sentences_subset.sentences_final.values
  for i, sentence in enumerate(sentences):
    if not crisis_sentences_subset['event_count'][i] in ["2 Codable Events","3 Codable Events"]:
      continue
    max_tokens=2700
    print(i, flush=True)
    print(sentence, flush=True)
    #Prompt1
    if crisis_sentences_subset['event_count'][i] in ["2 Codable Events"]:
      question="The sentence contains two distinct codable events. An event can be an action, a communication, or a thought. What are the two distinct events described in the sentence?"
    if crisis_sentences_subset['event_count'][i] in ["3 Codable Events"]:
      question="The sentence contains three distinct codable events. An event can be an action, a communication, or a thought. What are the three distinct events described in the sentence?"
    crisis_sentences_subset.loc[i,'event_count_question']=question
    prompt1="""USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence\n%s\n### End Question\n### Begin Debate (Carefully reread the specific sentence and the question. Write a short paragaph debating the merrits of different possible answers. Be factually correct and very concise.)\nASSISTANT: """ % (story, sentence, question)
    tostrip= round(min(max_tokens - 300 - generator.tokenizer.encode(prompt1).shape[1],0)*4)
    if tostrip:
      prompt1="""USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence\n%s\n### End Question\n### Begin Debate (Carefully reread the specific sentence and the question. Write a short paragaph debating the merrits of different possible answers. Be factually correct and very concise.)\nASSISTANT: """ % (story[:tostrip], sentence, question)
    output_debate = generator.generate_simple_rex( prompt1, max_new_tokens = 300, custom_stop= "#")
    output_debate = output_debate.replace(prompt1, "").strip().split("\n\n")[0].strip() #note I slightly changed the strips here
    crisis_sentences_subset.loc[i,'event_split_debate']=output_answer_review
    print(output_debate, flush=True)
    #Prompt2
    prompt2 = """USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence\n%s\n### End Question\n### Begin Debate (Carefully reread the specific sentence and the question. Write a short paragaph debating the merrits of different possible answers. Be factually correct and very concise.)\n%s\n### End Debate\n### Begin List of Events (write each event out as a full complete sentence that repeats all of the relevant actors and information. Repeat all relevant information of the original sentence such as dates and actors. Place each new sentence on a numbered line.)\nASSISTANT: """ % (story, sentence, question, output_debate)
    tostrip= round(min(max_tokens - 300 - generator.tokenizer.encode(prompt2).shape[1],0)*4)
    if tostrip:
      prompt2 = """USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence\n%s\n### End Question\n### Begin Debate (Carefully reread the specific sentence and the question. Write a short paragaph debating the merrits of different possible answers. Be factually correct and very concise.)\n%s\n### End Debate\n### Begin List of Events (write each event out as a full complete sentence that repeats all of the relevant actors and information. Repeat all relevant information of the original sentence such as dates and actors. Place each new sentence on a numbered line.)\nASSISTANT: """ % (story[:tostrip], sentence, question, output_debate)
    output_answer =  generator.generate_simple_rex(prompt2, max_new_tokens = 300, custom_stop= "#")
    output_answer = output_answer.replace(prompt2, "").strip().split("\n\n")[0].strip() #note I slightly changed the strips here
    crisis_sentences_subset.loc[i,'event_split_output_final_answer']=output_answer
    print(output_answer, flush=True)

  crisis_sentences_subset['event_split_prompt_for_saving']=prompt2
  crisis_sentences_subset.to_csv(fileout, index=False)

