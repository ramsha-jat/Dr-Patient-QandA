from flask import Flask, request, render_template, session, redirect, url_for
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import logging



app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

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
        transcript = request.form['transcript']
        session['transcript'] = transcript
        session['qa'] = []  # Initialize the list to store questions and answers
        return redirect(url_for('ask_question'))
    return render_template('index.html')

@app.route('/ask', methods=['GET', 'POST'])
def ask_question():
    transcript = session.get('transcript')
    if not transcript:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        question = request.form['question']
        answer = answer_question(question, transcript)
        
        # Add question and answer to the session
        qa_list = session.get('qa', [])
        qa_list.append({'question': question, 'answer': answer})
        session['qa'] = qa_list
        
        return render_template('results.html', qa_list=qa_list)
    
    return render_template('ask.html')

if __name__ == '__main__':
    app.run(debug=True)
