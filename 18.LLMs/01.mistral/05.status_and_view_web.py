import os
import json
import glob
import pandas as pd
from flask import Flask, render_template_string

# Paths
EXTRACTED_DIR = r'D:/OEssam/extracted_json'
CHROMA_DB_DIR = r'D:/OEssam/chroma_db'
TUNING_DIR = r'C:/Users/Omar Essam2/OneDrive - Rowad Modern Engineering/x004 Data Science/03.rme.db/05.llm/gpt.tuning'
TUNING_FILE = os.path.join(TUNING_DIR, 'gpt_feedback_log.jsonl')

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RME LLM Data Status & Q&A Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f6fb; margin: 0; padding: 0; }
        .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px #0001; padding: 32px; }
        h1 { text-align: center; color: #333; }
        h2 { color: #2563eb; margin-top: 32px; }
        table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #e0e7ff; }
        tr:nth-child(even) { background: #f9fafb; }
        .warn { color: #b91c1c; font-weight: bold; }
        .info { color: #2563eb; }
        .small { font-size: 0.95em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>RME LLM Data Status & Q&amp;A Viewer</h1>
        <h2>Extracted JSONs</h2>
        {% if extracted_jsons %}
            <ul>
            {% for f in extracted_jsons %}
                <li>{{ f.name }} <span class="small">({{ f.size }} KB)</span></li>
            {% endfor %}
            </ul>
        {% else %}
            <div class="warn">[WARN] Extracted JSON directory not found or empty.</div>
        {% endif %}
        <h2>Chroma DB</h2>
        {% if chroma_db %}
            <ul>
            {% for f in chroma_db %}
                <li>{{ f }}</li>
            {% endfor %}
            </ul>
        {% else %}
            <div class="warn">[WARN] Chroma DB directory not found or empty.</div>
        {% endif %}
        <h2>GPT Training Data</h2>
        {% if qa_count > 0 %}
            <div class="info">Found {{ qa_count }} Q&amp;A pairs in training data.</div>
            <table>
                <tr><th>#</th><th>Question</th><th>Answer</th></tr>
                {% for i, row in qa_table.iterrows() %}
                <tr>
                    <td>{{ i+1 }}</td>
                    <td style="max-width:350px; word-break:break-word;">{{ row['question'] }}</td>
                    <td style="max-width:500px; word-break:break-word;">{{ row['answer'] }}</td>
                </tr>
                {% endfor %}
            </table>
        {% else %}
            <div class="warn">[WARN] GPT training data file not found or empty.</div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    # Extracted JSONs
    extracted_jsons = []
    if os.path.exists(EXTRACTED_DIR):
        json_files = glob.glob(os.path.join(EXTRACTED_DIR, '*.json'))
        for f in json_files:
            extracted_jsons.append({'name': os.path.basename(f), 'size': os.path.getsize(f)//1024})
    # Chroma DB
    chroma_db = []
    if os.path.exists(CHROMA_DB_DIR):
        chroma_db = os.listdir(CHROMA_DB_DIR)
    # GPT Training Data
    qa_count = 0
    qa_table = pd.DataFrame()
    if os.path.exists(TUNING_FILE):
        with open(TUNING_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        qa_count = len(lines)
        if qa_count > 0:
            qa_list = [json.loads(line) for line in lines]
            qa_table = pd.DataFrame(qa_list)
            if not qa_table.empty:
                qa_table = qa_table[['question', 'answer']].head(50)  # Show up to 50 Q&A
    return render_template_string(HTML,
        extracted_jsons=extracted_jsons,
        chroma_db=chroma_db,
        qa_count=qa_count,
        qa_table=qa_table
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True) 