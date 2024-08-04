from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import logging

# Disable warnings
logging.disable(logging.WARNING)

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, context, max_length=512):
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    )

    input_ids = inputs["input_ids"].tolist()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        transcript_file = request.files['transcript']
        questions = request.form.getlist('questions')

        # Read the transcript
        transcript = transcript_file.read().decode('utf-8')

        # Get answers for the questions
        answers = {}
        for question in questions:
            answer = answer_question(question, transcript)
            answers[question] = answer

        return render_template('results.html', transcript=transcript, questions=questions, answers=answers)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
