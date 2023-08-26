story= "Bolshevik Russia, precursor to the Soviet Union The Communist regime, which had attained power in Russia on 7 November 1917, opted to withdraw from World War I through a separate peace with Germany--the Treaty of Brest-Litovsk on 3 March 1918."
coding_instructions = "Given the story above, classify the sentence below as either an action or an interaction. An action involves only one actor, and an interaction involves two or more actors."
#sentence="They then signed a different treaty in 1980."
sentence="Russia fortified its border."
prompt_template= story + "\n" + coding_instructions + "\n" + sentence + "\n\n### Response:"
tokens = tokenizer(prompt_template, return_tensors="pt").to("cuda:0").input_ids
output = model.generate(input_ids=tokens, max_new_tokens=3, do_sample=True, temperature=0.8)
print(tokenizer.decode(output[0]))
