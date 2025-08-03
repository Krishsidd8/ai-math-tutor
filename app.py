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
from model_definitions import OCRModel, tokenizer, transform, predict
from config import device, checkpoint_path

# ======== Flask Setup =============== #
app = Flask(__name__)
CORS(app)

# ======== Load Model Once ============ #
model = OCRModel(len(tokenizer.vocab)).to(device)
state = torch.load(checkpoint_path, map_location=device)
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

# ======== Home Route ============ #
@app.route("/", methods=["GET"])
def home():
    return "AI Math Tutor API is running. Use the /solve endpoint."

# ======== Solve Route =========== #
@app.route("/solve", methods=["POST"])
def solve_equation():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        image = Image.open(io.BytesIO(request.files['image'].read())).convert('L')

        # 🔽 Optional: downsample to reduce memory usage
        image = image.resize((512, 384))

        # ✅ Predict LaTeX
        latex = predict(image, model, tokenizer.t2i)

        # ✅ Solve using SymPy
        steps = get_sympy_steps(latex)

        # 🧹 Free memory
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