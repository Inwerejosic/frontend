from flask import Flask, render_template, request, jsonify
# ... (rest of your existing Flask code from previous responses) ...
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) # debug=True for development


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
# MODEL_NAME = "google/flan-t5-base"
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
