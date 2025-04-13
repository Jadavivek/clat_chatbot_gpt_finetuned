import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_response(query):
    input_text = f"Question: {query}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=150, temperature=0.7, no_repeat_ngram_size=2)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

st.title("CLAT GPT Chatbot (Fine-Tuned)")
user_input = st.text_input("Ask your CLAT query:")

if user_input:
    st.write("Bot:", generate_response(user_input))
