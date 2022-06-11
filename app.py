from venv import create
from flask import Flask, render_template, jsonify,request
from creacion_del_indice import *
import psycopg2

conn = psycopg2.connect(host="localhost",database="proyecto2_base_de_datos_2",user = "postgres" ,password = "1234", options = "-c search_path=public")


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
    cr = conn.cursor()


    list_tokens = text.split(" ")
    query_text_postgres = list_tokens[0] 
    list_tokens = list_tokens[1:]

    for palabra in list_tokens:
        query_text_postgres += ' | ' + palabra  


    query_string = """SELECT id, texto, ts_rank_cd(weighted_tsv, query) AS rank
                FROM tweets, plainto_tsquery('english', \'""" + query_text_postgres +  """\') query
                WHERE query @@ weighted_tsv and id < 1009
                ORDER BY rank DESC
                LIMIT """ + k  +  """;"""

    print(query_string)
    cr.execute(query_string)
    query_postgres_result = [(i[2],i[0],i[1]) for i in cr.fetchall()] 
    print(query_postgres_result)
    cr.close()

    return render_template('topk.html', tweets = tweets, tweets_postgres = query_postgres_result)


if __name__ == '__main__':
    app.run('0.0.0.0', port = 5000, debug = True)
    conn.close()