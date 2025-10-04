from flask import Flask, render_template, request, jsonify
from trained_model import classify_message
import os

app = Flask(__name__)

@app.route("/", methods=["POST"])
def home():
    if request.is_json:
        request_object = request.get_json(silent=True) or {}
        input = request_object.get("message")
    else:
        input = request.form.get("message")

    if not input:
        return jsonify({"error": "no message provided (expected 'message' field)"}), 400

    try:
        label, conf, probs = classify_message(input, threshold=0.70, return_probs=True)
    except Exception as e:
        return jsonify({"error": "classification failed", "detail": str(e)}), 500

    return jsonify({
        "label": label,
        "confidence": conf,
        "probs": probs
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # set debug=False for production
    print("server is running")
    app.run(debug=True, host="0.0.0.0", port=port)



