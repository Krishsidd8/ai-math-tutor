from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import sympy as sp
from sympy import Eq, simplify, factor, solve
from sympy.parsing.latex import parse_latex
from flask_cors import CORS
import gc

# ======== Import Support Code ============ #
from model_definitions import OCRModel, transform, predict, load_tokenizer
from config import device, checkpoint_path, tokenizer_path
import os
import urllib.request

# ======== Flask Setup =============== #
app = Flask(__name__)
CORS(app)

# ======== Load Tokenizer & Model Once ============ #
tokenizer = load_tokenizer(tokenizer_path)

model = OCRModel(len(tokenizer.vocab)).to(device)
if not os.path.exists(checkpoint_path):
    print("Downloading model checkpoint...")
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=1IttUFMaSxgyEbunjwvnOntFtcWWWuQdh",
        checkpoint_path
    )
    print("Model downloaded successfully.")
state = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(state['model'])
model.eval()

# ======== SymPy-Based Math Steps ==== #
def get_sympy_steps(latex_input):
    try:
        expr = parse_latex(latex_input)
        if not isinstance(expr, sp.Equality):
            expr = Eq(expr, 0)

        lhs, rhs = expr.lhs, expr.rhs
        symbols = list(expr.free_symbols)
        if not symbols:
            raise ValueError("No variable found in equation.")
        var = symbols[0]

        steps = []

        # Step 1: Move terms to one side
        full_expr = simplify(lhs - rhs)
        steps.append({
            "step": "Move all terms to one side",
            "symbolic": f"{sp.latex(full_expr)} = 0"
        })

        # Step 2: Try factoring
        factored = factor(full_expr)
        if factored != full_expr:
            steps.append({
                "step": "Factor the expression",
                "symbolic": f"{sp.latex(factored)} = 0"
            })
        else:
            steps.append({
                "step": "Expression cannot be factored further",
                "symbolic": f"{sp.latex(full_expr)} = 0"
            })

        # Step 3: Solve
        solutions = solve(expr, var)
        for sol in solutions:
            steps.append({
                "step": f"Solve for {var}",
                "symbolic": f"{sp.latex(var)} = {sp.latex(sol)}"
            })

        return steps

    except Exception as e:
        raise ValueError(f"SymPy failed to solve equation: {str(e)}")

# ======== Solve Route =========== #
@app.route("/solve", methods=["POST"])
def solve_equation():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        image = Image.open(io.BytesIO(request.files['image'].read())).convert('L')
        image = image.resize((512, 384))

        with torch.no_grad():
            latex = predict(image, model, tokenizer)

        steps = get_sympy_steps(latex)

        del image
        gc.collect()
        torch.cuda.empty_cache()

        return jsonify({
            "latex": latex,
            "steps": steps
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask import send_from_directory

@app.route("/docs/<path:filename>")
def serve_docs(filename):
    return send_from_directory("docs", filename)

@app.route("/")
def serve_index():
    return send_from_directory("docs", "index.html")
