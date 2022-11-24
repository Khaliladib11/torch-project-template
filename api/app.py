from flask import Flask, request, jsonify, abort
from inference import inference

app = Flask(__name__)

@app.get('/')
def welcome():
    return "Hello World from Flask"


# prediction REST API
@app.route('/api/predictions', methods=['POST'])
def pred():
    if request.method == 'POST':
        input = None  # read the input from the requets objects...

        out = inference(input)

        return jsonify({'msg': 'success', 'predcition': out}), 200
    else:
        abort(400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)