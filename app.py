from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import sympy as sp
from sympy import Eq, simplify, factor, solve
from sympy.parsing.latex import parse_latex
from flask_cors import CORS

# ======== Import Support Code ============ #
from model_definitions import OCRModel, tokenizer, transform, predict
from config import device, checkpoint_path

# ======== Flask Setup =============== #
app = Flask(__name__)
CORS(app)

# ======== SymPy-Based Math Steps ==== #
def get_sympy_steps(latex_input):
    try:
        expr = parse_latex(latex_input)
        if not isinstance(expr, sp.Equality):
            expr = Eq(expr, 0)

        lhs, rhs = expr.lhs, expr.rhs
        var = list(expr.free_symbols)[0]

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

# ======== API Route ================= #
@app.route("/solve", methods=["POST"])
def solve_equation():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        image = Image.open(io.BytesIO(request.files['image'].read())).convert('L')

        # ======== Lazy Load Model ============ #
        model = OCRModel(len(tokenizer.vocab)).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model'])
        model.eval()

        # ======== Prediction and Processing === #
        latex = predict(image, model, tokenizer.t2i)
        steps = get_sympy_steps(latex)

        # ======== Free Memory ============ #
        del model, image, state
        torch.cuda.empty_cache()

        return jsonify({
            "latex": latex,
            "steps": steps
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ Run =================== #
if __name__ == "__main__":
    app.run(debug=True)
