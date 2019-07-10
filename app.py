from flask import Flask, request, render_template, jsonify

app = Flask(__name__, static_url_path="")

@app.route("/")
def index():
    """Return the main page."""
    return render_template("index.html")


@app.route("/output", methods=["GET", "POST"])
def output():
    """Retun text from user input"""
    data = request.get_json(force=True)
    # every time the user_input identifier
    print(data)
    #multiply input by 20
    output = float(data["text_box"]) * 20
    #print contents of output to terminal
    print(output)
    return jsonify(output)