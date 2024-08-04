# Flask Q&A Application

This is a simple Flask web application that allows users to upload a conversation transcript and then ask questions about the transcript one by one. The application uses a pre-trained BERT model to answer the questions based on the uploaded transcript.

## Features

- Upload a conversation transcript.
- Ask questions about the transcript.
- Get answers from the pre-trained BERT model.
- View all previously asked questions and their answers.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/flask-qa-app.git
    cd flask-qa-app
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    **Note:** If `requirements.txt` is not provided, create it with the following content:
    
    ```
    Flask==2.0.1
    torch==1.9.0
    transformers==4.9.1
    ```

4. **Run the Flask application:**

    ```bash
    python app.py
    ```

5. **Open your web browser and go to:**

    ```
    http://127.0.0.1:5000/
    ```

## Usage

1. **Upload Transcript:**

    - Paste the conversation transcript into the text area on the main page and click "Submit".

2. **Ask Questions:**

    - After submitting the transcript, you will be redirected to a page where you can ask questions one by one.
    - Type your question into the input field and click "Submit".
    - The application will display the answer and show all previously asked questions and answers.

3. **Upload a New Transcript:**

    - If you want to upload a new transcript, click the "Upload a new transcript" link.

## Project Structure

