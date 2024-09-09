# language_models/dialogpt_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DialoGPTModel:
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None

    def generate_response(self, user_input, temperature=0.8, top_k=25, top_p=1.6):
        # Tokenize the user's input
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Add the user's input to the chat history
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Create the attention mask
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

        # Generate the bot's response
        self.chat_history_ids = self.model.generate(
            bot_input_ids, 
            max_length=1000, 
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
            temperature=temperature,  # A higher value (e.g., 0.9) makes the responses more varied and creative, while a lower value (e.g., 0.7) makes the responses more deterministic.
            top_k=top_k,  # A lower value (e.g., 50) limits the number of tokens considered for response generation, making the responses more coherent.
            top_p=top_p,  # A lower value (e.g., 0.9) considers only the tokens whose cumulative probability is below p, making the responses more coherent.
            do_sample=True  # Enable sampling
        )

        # Decode the bot's response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

# Test the DialoGPT model with a simple conversation
if __name__ == "__main__":
    dialogpt_model = DialoGPTModel()
    user_input = "Hello, how are you?"
    response = dialogpt_model.generate_response(user_input)
    print("User:", user_input)
    print("Bot:", response)
    print()

    user_input = "What is your name?"
    response = dialogpt_model.generate_response(user_input)
    print("User:", user_input)
    print("Bot:", response)
    print()

    user_input = "What is the meaning of life?"
    response = dialogpt_model.generate_response(user_input)
    print("User:", user_input)
    print("Bot:", response)
    print()
