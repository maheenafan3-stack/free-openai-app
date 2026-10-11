from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = Flask(__name__)
CORS(app)

# Load a free, open-source model
# Using DialoGPT (small version for faster inference)
MODEL_NAME = "microsoft/DialoGPT-medium"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set model to evaluation mode
model.eval()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded on {device}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '').strip()
    
    if not user_input:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        # Tokenize input
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
        
        # Generate response
        chat_history_ids = model.generate(
            new_input_ids,
            max_length=200,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode response
        response = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return jsonify({
            'response': response,
            'user_input': user_input
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'current_model': MODEL_NAME,
        'available_models': [
            'microsoft/DialoGPT-small',
            'microsoft/DialoGPT-medium',
            'microsoft/DialoGPT-large',
            'gpt2',
            'distilgpt2'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
