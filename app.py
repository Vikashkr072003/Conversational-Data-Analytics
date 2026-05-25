import os
import io
import base64
import contextlib

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template
from groq import Groq
from dotenv import load_dotenv

# ── Initialisation ────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df: pd.DataFrame | None = None   # Global dataset holder


# ── Helper ────────────────────────────────────────────────────────────────────
def _read_file(path: str, skip: int = 0) -> pd.DataFrame:
    """Read CSV or Excel; fall back to skip=0 if skip rows cause an error."""
    is_csv = path.lower().endswith('.csv')
    try:
        return pd.read_csv(path, skiprows=skip) if is_csv else pd.read_excel(path, skiprows=skip)
    except Exception:
        return pd.read_csv(path) if is_csv else pd.read_excel(path)


def _ffill_if_exists(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Forward-fill only columns that actually exist in the dataframe."""
    for col in cols:
        if col in frame.columns:
            frame[col] = frame[col].ffill()
    return frame


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global df

    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file provided."}), 400

    # Validate extension
    allowed = {'.csv', '.xlsx', '.xls'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use CSV or Excel."}), 400

    # Save
    folder = app.config['UPLOAD_FOLDER']
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)
    file.save(file_path)

    # Load & clean
    try:
        df = _read_file(file_path, skip=2)
        df = _ffill_if_exists(df, ['Zone', 'State'])
        df.columns = df.columns.str.strip()   # strip accidental whitespace from headers
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 500

    preview = {
        "message": "File uploaded successfully!",
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
    }
    return jsonify(preview)


@app.route('/analyze', methods=['POST'])
def analyze():
    global df

    if df is None:
        return jsonify({"error": "No dataset uploaded yet. Please upload a file first."}), 400

    body = request.get_json(silent=True) or {}
    user_query = body.get('query', '').strip()
    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # ── Build LLM prompt ──────────────────────────────────────────────────────
    code_prompt = f"""
You are a senior Python data analyst. A pandas DataFrame called 'df' is already loaded in memory.

Column names  : {df.columns.tolist()}
Shape         : {df.shape}
Sample (3 rows): {df.head(3).to_dict(orient='records')}

User query: "{user_query}"

STRICT RULES — follow every one:
1. Write only raw Python code. No markdown fences, no backticks, no prose.
2. Answer the query numerically/textually AND print the result with print().
3. ALWAYS create one matplotlib chart that best visualises the answer:
   - Call plt.figure(figsize=(8, 5)) to start a new figure.
   - Set a descriptive plt.title(), plt.xlabel(), plt.ylabel() where applicable.
   - Call plt.tight_layout() at the end.
   - Do NOT call plt.show() or plt.savefig().
4. Available names: df, pd, plt only.
5. Never use a variable named 'it'.
6. Handle missing/NaN values gracefully (e.g. dropna before plotting).
"""

    try:
        plt.close('all')   # wipe any leftover figures

        # ── Call Groq ─────────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": code_prompt}],
            temperature=0.2,
        )
        raw_code: str = response.choices[0].message.content

        # Strip markdown fences the model sometimes still adds
        generated_code = (
            raw_code
            .replace('```python', '')
            .replace('```', '')
            .strip()
        )

        # ── Execute generated code ────────────────────────────────────────────
        stdout_capture = io.StringIO()
        exec_namespace = {'df': df, 'pd': pd, 'plt': plt}

        with contextlib.redirect_stdout(stdout_capture):
            exec(generated_code, exec_namespace)   # single namespace → no scoping bugs

        text_result = stdout_capture.getvalue().strip()

        # ── Capture chart ─────────────────────────────────────────────────────
        img_base64 = ""
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
            buf.seek(0)                            # ← must reset before reading
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

        plt.close('all')   # clean up after capturing

        return jsonify({
            "code":   generated_code,
            "result": text_result or "Analysis complete.",
            "image":  img_base64,
        })

    except Exception as e:
        plt.close('all')
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)