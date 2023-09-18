import flask
from flask import Flask, request, render_template, jsonify
from chat import  get_response

app = Flask(__name__)

@app.get("/")  # request type get
def index_get():
    return render_template("base.html")

@app.post("/predict") # request type post
def predict():
    text = request.get_json().get("message")
    #TODO: check if text is valid
    response = get_response(text)
    message = {"answer":response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)