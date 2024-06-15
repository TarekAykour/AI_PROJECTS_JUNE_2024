from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model name
model_name = "facebook/blenderbot-400M-distill"


# Load model (download on first run and reference local installation for consequent runs)
"""

model is an instance of the class AutoModelForSeq2SeqLM, which allows you to interact with your chosen language model.
tokenizer is an instance of the class AutoTokenizer,
which optimizes your input and passes it to the language model efficiently. 
It does so by converting your text input to “tokens”, 
 which is how the model interprets the text.

"""
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# tracking convo history
conversation_history = []


"""
The transformers library function you are using expects to receive the conversation history as a string,
 with each element separated by the newline character '\n'. 
Thus, you create such a string.

You'll use the join() method in Python to do exactly that. 
(Initially, your history_string will be an empty string, which is okay, 
and will grow as the conversation goes on).
"""

history_string = "\n".join(conversation_history)


input_text ="hello, how are you doing?"


inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
# print(inputs)

tokenizer.pretrained_vocab_files_map


outputs = model.generate(**inputs)
# print(outputs)

"""
You may decode the output using tokenizer.decode(). 
This is known as "detokenization" or "reconstruction". 
It is the process of combining or merging individual tokens back into their original form,
 to reconstruct the original text or sentence.
"""
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)



conversation_history.append(input_text)
conversation_history.append(response)
# print(conversation_history)


while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
