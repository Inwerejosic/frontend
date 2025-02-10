# # from flask import Flask, render_template, request, jsonify
# # # ... (rest of your existing Flask code from previous responses) ...
# # app = Flask(__name__)

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # if __name__ == '__main__':
# #     app.run(debug=True) # debug=True for development


# # prompt: want to run this from azure vm

# import os
# import torch
# import random
# from flask import Flask, request, jsonify, render_template
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# # Define the criteria and weights
# criteria = ["is_safe_enough", "reliab_center_m", "skilled_and_trained", "cost_optimized"]
# weights = torch.tensor([0.41, 0.26, 0.19, 0.14])

# # Load Hugging Face model
# MODEL_NAME = "google/flan-t5-small"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# # Predefined recommendation templates
# templates = [
#     "I highly recommend {alternative} due to its outstanding balance of factors.",
#     "{alternative} is the best choice considering all important criteria.",
#     "Based on the analysis, {alternative} stands out as the optimal solution.",
#     "For efficiency and reliability, {alternative} is the most suitable option.",
#     "Given the project constraints, {alternative} emerges as the top recommendation.",
#     "{alternative} has been identified as the most effective solution.",
# ]

# # Initialize Flask App
# app = Flask(__name__, template_folder='templates')

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/make_decision', methods=['POST'])
# def make_decision():
#     try:
#         data = request.form.to_dict()
#         alternatives = []
#         for i in range(1, 5):
#             name = data.get(f'alternative_name_{i}')
#             if name:
#                 values = []
#                 for j in range(len(criteria)):
#                     value = data.get(f'criteria_{j+1}_alt_{i}')
#                     if value is None or not value.isdigit():
#                         return jsonify({'error': 'Invalid input. Please enter numerical values for all criteria'}), 400
#                     values.append(float(value))
#                 alternatives.append({'name': name, 'values': values})

#         if not alternatives:
#             return jsonify({'error': 'No alternatives provided.'}), 400

#         names = [alt['name'] for alt in alternatives]
#         scores = [alt['values'] for alt in alternatives]

#         scores_tensor = torch.tensor(scores, dtype=torch.float32)
#         weighted_sums = torch.matmul(scores_tensor, weights)
#         best_index = torch.argmax(weighted_sums).item()
#         best_alternative = names[best_index]

#         ai_prompt = f"Give a professional recommendation for selecting {best_alternative}."
#         ai_generated_text = generator(ai_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
#         random_template = random.choice(templates).format(alternative=best_alternative)
#         final_recommendation = f"{random_template} {ai_generated_text}"

#         return render_template('result.html', selected_alternative=best_alternative, recommendation=final_recommendation)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#      port = int(os.environ.get('PORT', 5000))  # Use PORT env or default to 5000
#     app.run(host='0.0.0.0', port=port)

# Import required libraries
import os
import torch
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
# from pyngrok import ngrok
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# **STEP 1: Set Your ngrok Auth Token**
# NGROK_AUTH_TOKEN = "2ACXPJrRiKEkqH6Z7upsuLqahKy_o4Kxspph37SS2azBA8eq"  # Replace with your actual token
# !ngrok config add-authtoken $NGROK_AUTH_TOKEN

# Define the criteria and weights
criteria = ["is_safe_enough", "reliab_center_m", "skilled_and_trained", "cost_optimized"]
weights = torch.tensor([0.41, 0.26, 0.19, 0.14])

# Load Hugging Face model
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Predefined recommendation templates
templates = [
    "I highly recommend {alternative} due to its outstanding balance of factors.",
    "{alternative} is the best choice considering all important criteria.",
    "Based on the analysis, {alternative} stands out as the optimal solution.",
    "For efficiency and reliability, {alternative} is the most suitable option.",
    "Given the project constraints, {alternative} emerges as the top recommendation.",
    "{alternative} has been identified as the most effective solution.",
]

# Initialize Flask App
app = Flask(__name__)
CORS(app)

@app.route('/make_decision', methods=['POST'])
def make_decision():
    try:
        # Receive JSON input from user
        data = request.get_json()
        if not isinstance(data, dict) or 'alternatives' not in data:
            return jsonify({'error': 'Invalid input format. Expected a dictionary with an "alternatives" key.'}), 400

        alternatives = data['alternatives']
        if not isinstance(alternatives, list) or not all(isinstance(alt, dict) and 'name' in alt and 'values' in alt for alt in alternatives):
            return jsonify({'error': 'Invalid "alternatives" format. Expected a list of dictionaries with "name" and "values" keys.'}), 400

        # Extract names and scores
        names = [alt['name'] for alt in alternatives]
        scores = [alt['values'] for alt in alternatives]

        # Ensure scores are numerical and correctly shaped
        try:
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
        except Exception as e:
            return jsonify({'error': f'Error processing scores: {str(e)}'}), 400

        # Compute weighted sums
        weighted_sums = torch.matmul(scores_tensor, weights)

        # Find the best alternative
        best_index = torch.argmax(weighted_sums).item()
        best_alternative = names[best_index]

        # Generate AI-based professional recommendation
        ai_prompt = f"Give a professional recommendation for selecting {best_alternative}."
        ai_generated_text = generator(ai_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']

        # Pick a random predefined template
        random_template = random.choice(templates).format(alternative=best_alternative)

        # Combine AI-generated and template-based recommendations
        final_recommendation = f"{random_template} {ai_generated_text}"

        return jsonify({'selected_alternative': best_alternative, 'recommendation': final_recommendation})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# # **Expose Flask via ngrok**
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)
# public_url = ngrok.connect(5000).public_url
# print(f"Flask App Running at: {public_url}")

# Start the Flask app
app.run(port=5000)
