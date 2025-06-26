from flask import Flask, render_template_string

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RME Financial Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f1f4f9, #dff2ff);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .card {
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 0 30px rgba(0,0,0,0.08);
            border-radius: 16px;
            background-color: white;
        }
        h2 {
            text-align: center;
            margin-bottom: 30px;
            color: #1F4E78;
        }
        .btn-lg {
            padding: 15px 20px;
            font-size: 1.2rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>📊 RME Financial Tools</h2>
        <a href="http://10.10.11.242:5000" class="btn btn-primary btn-lg">
            <i class="bi bi-file-earmark-text"></i> Receipts Report
        </a>
        <a href="http://10.10.11.242:5055" class="btn btn-success btn-lg">
            <i class="bi bi-diagram-3"></i> Sector Summary
        </a>
        <a href="http://10.10.11.242:5001" class="btn btn-info btn-lg text-white">
            <i class="bi bi-cash-stack"></i> SWD Collections
        </a>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
