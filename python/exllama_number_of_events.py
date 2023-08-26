
#About 20 minutes per crisis. 6.8 days end to end.
#made it to 71. Keep going when free time.

# 
# story="""The Cuban Missile Crisis, also known as the October Crisis (of 1962) (Spanish: Crisis de Octubre) in Cuba, the Caribbean Crisis (Russian: Карибский кризис, tr. Karibsky krizis, IPA: [kɐˈrʲipskʲɪj ˈkrʲizʲɪs]) in Russia, or the Missile Scare, was a 13-day (October 16 – October 29, 1962) confrontation between the United States and the Soviet Union, which escalated into an international crisis when American deployments of missiles in Italy and Turkey were matched by Soviet deployments of similar ballistic missiles in Cuba. Despite the short time frame, the Cuban Missile Crisis remains a defining moment in American national security and nuclear war preparation. The confrontation is often considered the closest the Cold War came to escalating into a full-scale conflict, nuclear war.[4]
# In 1961, the US government put Jupiter nuclear missiles in Italy and Turkey. It had also trained a paramilitary force of Cuban exiles, which the CIA led in an attempt to invade Cuba and overthrow its government. Starting in November of that year, the US government engaged in a campaign of terrorism and sabotage in Cuba, referred to as the Cuban Project, which continued throughout the first half of the 1960s. The Soviet administration was concerned about a Cuban drift towards China, with which the Soviets had an increasingly fractious relationship. In response to these factors, Soviet First Secretary, Nikita Khrushchev, agreed with the Cuban Prime Minister, Fidel Castro, to place nuclear missiles on the island of Cuba to deter a future invasion. An agreement was reached during a secret meeting between Khrushchev and Castro in July 1962, and construction of a number of missile launch facilities started later that summer.
# During the campaigning for the 1962 United States elections, the White House denied the charges for months and ignored the presence of Soviet missiles positioned approximately 90 mi (140 km) away from Florida. Later, the missile preparations were confirmed when a US Air Force U-2 spy plane produced clear photographic evidence of medium-range R-12 (NATO code name SS-4) and intermediate-range R-14 (NATO code name SS-5) ballistic missile facilities.
# When this was reported to President John F. Kennedy, he then convened a meeting of the nine members of the National Security Council and five other key advisers, in a group that became known as the Executive Committee of the National Security Council (EXCOMM). During this meeting, President Kennedy was originally advised to carry out an air strike on Cuban soil in order to compromise Soviet missile supplies, followed by an invasion of the Cuban mainland. After careful consideration, President Kennedy chose a less aggressive course of action, in order to avoid a declaration of war. After consultation with EXCOMM, Kennedy ordered a naval "quarantine" on October 22 to prevent further missiles from reaching Cuba.[5] By using the term "quarantine", rather than "blockade" (an act of war by legal definition), the United States was able to avoid the implications of a state of war.[6] The US announced it would not permit offensive weapons to be delivered to Cuba and demanded that the weapons already in Cuba be dismantled and returned to the Soviet Union.
# After several days of tense negotiations, an agreement was reached between Kennedy and Khrushchev: publicly, the Soviets would dismantle their offensive weapons in Cuba and return them to the Soviet Union, subject to United Nations verification, in exchange for a US public declaration and agreement to not invade Cuba again. Secretly, the United States agreed with the Soviets that it would dismantle all of the Jupiter MRBMs which had been deployed to Turkey against the Soviet Union. There has been debate on whether or not Italy was included in the agreement as well. While the Soviets dismantled their missiles, some Soviet bombers remained in Cuba, and the United States kept the naval quarantine in place until November 20, 1962.[6]
# When all offensive missiles and the Ilyushin Il-28 light bombers had been withdrawn from Cuba, the blockade was formally ended on November 20. The negotiations between the United States and the Soviet Union pointed out the necessity of a quick, clear, and direct communication line between the two superpowers. As a result, the Moscow–Washington hotline was established. A series of agreements later reduced US–Soviet tensions for several years, until both parties eventually resumed expanding their nuclear arsenals.
# The compromise embarrassed Khrushchev and the Soviet Union because the withdrawal of US missiles from Italy and Turkey was a secret deal between Kennedy and Khrushchev, and the Soviets were seen as retreating from circumstances that they had started. Khrushchev's fall from power two years later was in part because of the Soviet Politburo's embarrassment at both Khrushchev's eventual concessions to the US and his ineptitude in precipitating the crisis in the first place. According to Dobrynin, the top Soviet leadership took the Cuban outcome as "a blow to its prestige bordering on humiliation".[7][8]"""
# 
# sentence="""After several days of tense negotiations, an agreement was reached between Kennedy and Khrushchev: publicly, the Soviets would dismantle their offensive weapons in Cuba and return them to the Soviet Union, subject to United Nations verification, in exchange for a US public declaration and agreement to not invade Cuba again."""
# 
# question="""question = {
# "Question Name": "Type of Event",
# "Question Description": "Count of codable events in this sentence? A codable event is where one or more actors commit an explicit action, speech, or thought as described in the sentence.",
# "Question Usage Notes": "Choose the number of distinct events first before coding the details of an event. If you change the number, it’ll reset what you’ve coded.",
# "Answer Options" : [
# {"Answer": "Background",
# "Definition": "Potentially useful context, but no codable actions from the crisis. Could refer to something before, during, or after the crisis.",
# "Positive Examples": ["There was no global organization at the time of this crisis.",
# "A crisis for the Soviet Union and Czechoslovakia over the Marshall Plan occurred between 3 and 11 July 1947."]
# },
# {"Answer": "No Action",
# "Definition": "No codable action because explicit statement of inaction. Actors refrained from doing something.",
# "Positive Examples": ["And violence was not used by any crisis actor.", "Kerr did not formulate proposals, but acted to moderate differences between the sides."]
# },
# {"Answer": "1 Codable Event",
# "Definition": "A single event. A single action. A single speech act, potentially about an action. A single thought, potentially about an act or speech.",
# "Positive Examples": ["The Dominican armed forces were put on alert.", "The next day Molotov demanded that Prague retract its acceptance."]
# },
# {"Answer": "2 Distinct Codable Events",
# "Definition": "Two distinct codable events.",
# "Positive Examples": ["Bolivia appealed to Argentina to send troops to assist in the counterinsurgency operation but received only supplies and arms.","This was followed by intense trilateral negotiations, involving New Delhi, Islamabad, and London, ending in a cease-fire agreement on 11 May."]
# },
# {"Answer": "3 Distinct Codable Events",
# "Definition": "Three distinct codable events.",
# "Positive Examples": ["Supported and protected by Soviet troops, the Tudeh occupied several government buildings and issued a manifesto demanding administrative and cultural autonomy for Azerbaijan.", "The U.S., on the other hand, changed from supporting the Netherlands to pressing them to sign the Renville Agreement, to securing Dutch compliance with UN resolutions by threatening to suspend U.S. aid."]
# }
# ]
# }"""


import numpy as np
generator=icbe_llm_generator()

import pandas as pd
icbe_llm_codebook = pd.read_csv("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/icbe_llm_codebook.csv")

df_codebook = icbe_llm_codebook[icbe_llm_codebook.Variable=='sentence_events'].drop('ICBE1Code', axis=1).reset_index(drop=True) #need to drop index

json_codebook = df_codebook.to_json(orient="records")

n=df_codebook.shape[0]
df_codebook.columns
df_dictionary={ 'Variable': df_codebook['Variable'][0],
  'Description': df_codebook['Description'][0],
  'Usage Notes': df_codebook['Usage Notes'][0],
  'Response Options':
    [{'Response':df_codebook['Response Option'][i],'Response Description':df_codebook['Response Description'][i],'Positive Examples':df_codebook['Positive Examples'][i]} for i in range(0,n)] #,'Negative Examples':df_codebook['Negative Examples'][i]
}
import json
df_json =json.dumps(df_dictionary)
question=df_json

prompt_for_saving="""USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence and the Available Answers (in Dictionary Format)\n%s\n### End Question\n### List the Question's Available Answers (print a list of the available answer options and nothing else) ASSISTANT:\n%s\nUSER: ### End List of Available Answers\n### Begin Debate (Carefully reread the specific sentence and the coding instructions of the question. Write a short paragaph debating the merrits of each response, conluding with a synthesis in favor of only one of them. Be factually correct and very concise.)\nASSISTANT: \n%s\nUSER: ### End Debate\n### Final Answer (Reread the Specific Sentence, the coding instructions of the question, and the debate and synthesize a single final answer.)\nASSISTANT:\n%s\nUSER: ### End Final Answer\n### Machine Readable Final Answer (Write the chosen answer on a new line by itself with no other text)\nASSISTANT:"""
def assemble_prompt(story, sentence, question, answer_review=None, debate=None, output_final_answer=None):
  prompt4="""USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence and the Available Answers (in Dictionary Format)\n%s\n### End Question\n### List the Question's Available Answers (print a list of the available answer options and nothing else) ASSISTANT:\n%s\nUSER: ### End List of Available Answers\n### Begin Debate (Carefully reread the specific sentence and the coding instructions of the question. Write a short paragaph debating the merrits of each response, conluding with a synthesis in favor of only one of them. Be factually correct and very concise.)\nASSISTANT: \n%s\nUSER: ### End Debate\n### Final Answer (Reread the Specific Sentence, the coding instructions of the question, and the debate and synthesize a single final answer.)\nASSISTANT:\n%s\nUSER: ### End Final Answer\n### Machine Readable Final Answer (Write the chosen answer on a new line by itself with no other text)\nASSISTANT:""" % (story, sentence, question,answer_review,debate,output_final_answer)
  prompt3="""USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence and the Available Answers (in Dictionary Format)\n%s\n### End Question\n### List the Question's Available Answers (print a list of the available answer options and nothing else) ASSISTANT:\n%s\nUSER: ### End List of Available Answers\n### Begin Debate (Carefully reread the specific sentence and the coding instructions of the question. Write a short paragaph debating the merrits of each response, conluding with a synthesis in favor of only one of them. Be factually correct and very concise.)\nASSISTANT: \n%s\nUSER: ### End Debate\n### Final Answer (Reread the Specific Sentence, the coding instructions of the question, and the debate and synthesize a single final answer.)\n""" % (story, sentence, question,answer_review,debate)
  prompt2="""USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence and the Available Answers (in Dictionary Format)\n%s\n### End Question\n### List the Question's Available Answers (print a list of the available answer options and nothing else) ASSISTANT:\n%s\nUSER: ### End List of Available Answers\n### Begin Debate (Carefully reread the specific sentence and the coding instructions of the question. Write a short paragaph debating the merrits of each response, conluding with a synthesis in favor of only one of them. Be factually correct and very concise.)\nASSISTANT: """ % (story, sentence, question,answer_review)
  prompt1="""USER: Carefully read the following text and be prepared to answer detailed questions about a specific sentence from it.\n### Begin Story\n%s\n### End Story\n### Begin Specific Sentence\n%s\n### End Specific Sentence\n### Begin Question about Sentence and the Available Answers (in Dictionary Format)\n%s\n### End Question\n### List the Question's Available Answers (print a list of the available answer options and nothing else) ASSISTANT:""" % (story, sentence, question)
  if answer_review==None and debate==None and output_final_answer==None:
    return(prompt1)
  if answer_review!=None and debate==None and output_final_answer==None:
    return(prompt2)
  if answer_review!=None and debate!=None and output_final_answer==None:
    return(prompt3)
  if answer_review!=None and debate!=None and output_final_answer!=None:
    return(prompt4)
# 
# prompt1=assemble_prompt(story, sentence, question, answer_review=None, debate=None)
# len(prompt1)/4
# output1 =  generator.generate_simple_rex(prompt1, max_new_tokens = 200, custom_stop= '#')
# output_answer_review = output1.replace(prompt1, "")
# prompt2=assemble_prompt(story, sentence, question, answer_review=output_answer_review, debate=None)
# len(prompt2)/4
# output2 =  generator.generate_simple_rex(prompt2, max_new_tokens = 200, custom_stop= '#') #Debate ran too long is what happened
# output_debate = output2.replace(prompt2, "")
# output_debate = output_debate.split("USER")[0].strip()
# prompt3=assemble_prompt(story, sentence, question, answer_review=output_answer_review, debate=output_debate)
# len(prompt3)/4
# output3 =  generator.generate_simple_rex(prompt3, max_new_tokens = 100, custom_stop= '#')
# output_final_answer = output3.replace(prompt3, "")
# prompt4=assemble_prompt(story, sentence, question, answer_review=output_answer_review, debate=output_debate, output_final_answer=output_final_answer)
# len(prompt4)/4
# output4 =  generator.generate_simple_rex(prompt4, max_new_tokens = 50, custom_stop= '#')
# output_final_answer_machine_readable = output4.replace(prompt4, "").split("USER")[0].strip()


#pip install pyread
import pyreadr
import pandas as pd
crisis_narratives = pyreadr.read_r("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/crises_narratives_rex_2023_webscrape.Rds").popitem()[1]
print(crisis_narratives.keys())
crisis_sentences = pyreadr.read_r("/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_in/crises_narratives_rex_2023_webscrape_parsed_sentences.Rds").popitem()[1]

#If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
for crisisnum in crisis_narratives.crisno.values:
  fileout="/media/skynet3/8tb_a/rwd_github_private/ICBeLLM/data_out/number_of_events/crisis_event_count_"+str(crisisnum)+".csv"
  if os.path.isfile(fileout):
    continue
  print("Starting crisis " +str(crisisnum))
  story=crisis_narratives.text[crisisnum-1] #remember to subtract off one
  story=story.split('References')[0].strip()
  crisis_sentences_subset = crisis_sentences[crisis_sentences.crisno==crisisnum].copy().reset_index(drop=True)
  crisis_sentences_subset['event_count']=''
  crisis_sentences_subset['event_count_question']=''
  crisis_sentences_subset['event_count_answer_review']=''
  crisis_sentences_subset['event_count_debate']=''
  crisis_sentences_subset['event_count_output_final_answer']=''

  sentences=crisis_sentences_subset.sentences_final.values
  for i, sentence in enumerate(sentences):
    if crisis_sentences_subset['chunks_class'][i]=="fragment":
      continue
    max_tokens=2700
    print(i, flush=True)
    print(sentence, flush=True)
    crisis_sentences_subset.loc[i,'event_count_question']=question
    prompt1=assemble_prompt(story, sentence, question, answer_review=None, debate=None) #trimming references off the story and neg examples from the question got us under 2700 but there are longer stories
    tostrip= round(min(max_tokens - 50 - generator.tokenizer.encode(prompt1).shape[1],0)*4)
    if tostrip:
      prompt1=assemble_prompt(story[:tostrip], sentence, question, answer_review=None, debate=None)
    #len(prompt1)/4
    output1 =  generator.generate_simple_rex(prompt1, max_new_tokens = 50, custom_stop= '#')
    output_answer_review = output1.replace(prompt1, "").split("\n\n")[0].strip()
    crisis_sentences_subset.loc[i,'event_count_answer_review']=output_answer_review
    print(output_answer_review, flush=True)
    prompt2=assemble_prompt(story, sentence, question, answer_review=output_answer_review, debate=None)
    tostrip= round(min(max_tokens - 200  - generator.tokenizer.encode(prompt2).shape[1],0)*4)
    if tostrip:
      prompt2=assemble_prompt(story[:tostrip], sentence, question, answer_review=output_answer_review, debate=None)
    #len(prompt2)/4
    output2 =  generator.generate_simple_rex(prompt2, max_new_tokens = 200, custom_stop= '#') #Debate ran too long is what happened
    output_debate = output2.replace(prompt2, "")
    output_debate = output_debate.split("USER")[0].strip()
    crisis_sentences_subset.loc[i,'event_count_debate']=output_debate   
    print(output_debate, flush=True)
    prompt3=assemble_prompt(story, sentence, question, answer_review=output_answer_review, debate=output_debate)
    tostrip= round(min(max_tokens - 50 - generator.tokenizer.encode(prompt3).shape[1],0)*4)
    if tostrip:
      prompt3=assemble_prompt(story[:tostrip], sentence, question, answer_review=output_answer_review, debate=output_debate)
    #len(prompt3)/4
    output3 =  generator.generate_simple_rex(prompt3, max_new_tokens = 50, custom_stop= '#')
    output_final_answer = output3.replace(prompt3, "")
    output_final_answer = output_final_answer.split("USER")[0].strip()
    crisis_sentences_subset.loc[i,'event_count_output_final_answer']=output_final_answer 
    print(output_final_answer, flush=True)
    prompt4=assemble_prompt(story, sentence, question, answer_review=output_answer_review, debate=output_debate, output_final_answer=output_final_answer)
    #len(prompt4)/4
    tostrip= round(min(max_tokens - 50 -  generator.tokenizer.encode(prompt4).shape[1],0)*4)
    if tostrip:
      prompt4=assemble_prompt(story[:tostrip], sentence, question, answer_review=output_answer_review, debate=output_debate, output_final_answer=output_final_answer)
    output4 =  generator.generate_simple_rex(prompt4, max_new_tokens = 50, custom_stop= '#')
    output_final_answer_machine_readable = output4.replace(prompt4, "").split("USER")[0].strip()
    crisis_sentences_subset.loc[i,'event_count']=output_final_answer_machine_readable 
    print(output_final_answer_machine_readable, flush=True)    
    print("\n", flush=True)
  
  crisis_sentences_subset['event_count_prompt_for_saving']=prompt_for_saving
  crisis_sentences_subset.to_csv(fileout, index=False)


