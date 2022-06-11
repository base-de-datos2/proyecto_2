from venv import create
from flask import Flask, render_template, jsonify,request
from creacion_del_indice import *


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query',methods=['GET'])
def query():
    text = request.args.get('query')
    k = request.args.get('k')
    keys, dict = create_tf_query(procesamiento(text))
    create_unit_vector_query('sorted_tokens.txt', 1009 -1 , keys, dict)
    tweets = topk(1009 -1, int(k))
    return render_template('topk.html', tweets = tweets)


if __name__ == '__main__':
    app.run('0.0.0.0', port = 5000, debug = True)