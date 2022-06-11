import pandas as pd
import nltk
from queue import PriorityQueue
import os
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import io
import numpy as np
from pathlib import Path

BLOCKSIZE = 4096


#nltk.download('punkt')
#nltk.download('stopwords')

stemmer = SnowballStemmer('english')



def procesamiento(texto):
    """
    Tokenizar texto
    """
    texto_tokens = nltk.word_tokenize(texto)

    stoplist = set(stopwords.words('english'))
    #with(open("stoplist.txt", encoding = 'latin-1')) as file:
        #stoplist = [line.lower().strip() for line in file]
    for element in [',', '.', '?', '¿','https',':','#','!','&', "'s", '--',' ','','@','*','$','(',')','%']:
        stoplist.add(element)

    texto_tokens_c = texto_tokens[ : ]
    for token in texto_tokens:
        if token.lower() in stoplist or '/' in token.lower():
            texto_tokens_c.remove(token)
    



    texto_tokens_s = []
    for w in texto_tokens_c:
        texto_tokens_s.append(stemmer.stem(w))
    return texto_tokens_s



def spimi(token_stream,str_tweet_id,local_dictionary):
    
    for term in token_stream:
            
        if local_dictionary.get(term): 
            exists = False
            coleccion = local_dictionary[term]
            for i in range(len(coleccion)):
                if coleccion[i][0] == str_tweet_id:
                    coleccion[i][1] += 1
                    exists = True
            if not exists:
                coleccion.append([str_tweet_id,1])
                
        else:
            local_dictionary[term] = [[str_tweet_id,1]]
    return local_dictionary


def convert_key_to_file_name(key):
    key_file = key.replace('/', '')
    key_file = key_file.replace('.', '')
    return key_file


def write_to_memory(sorted_keys, local_dictionary):
    with open('temp.txt', 'a') as file_1:
        file_1.write(str(len(sorted_keys)) + " 1" + '\n')
        for key in sorted_keys:
            file_1.write(key + '\n')
            key_file = convert_key_to_file_name(key)
            total_docs  = 0
            path_collection = 'collections/' + 'list_' + key_file + '.txt'
            Path(path_collection).touch(exist_ok = True)
              
            with open(path_collection,'r+') as postings_list:
                postings_list.seek(0,io.SEEK_END)
                if postings_list.tell() == 0:
                    for i in range(1000):
                        postings_list.write(' ')
                    postings_list.write('\n')
                else:
                    postings_list.seek(0)
                    total_docs = int(postings_list.readline().split(' ')[0])

                string_postings_list = ""
                for pair in [docid + '-' + str(tf).zfill(3) + '\n' for (docid,tf) in local_dictionary[key]]:
                    string_postings_list += pair
                    total_docs += 1

                postings_list.seek(0)
                for i in range(1000):
                    postings_list.write(' ')
                postings_list.seek(0)
                postings_list.write(str(total_docs))
                postings_list.seek(0,io.SEEK_END)
                postings_list.write(string_postings_list)  


def get_next_small_block(block_file, lines_per_block):
    '''
    Lee un bloque pequeño y retorna la posicion en el archivo para uso futuro
    '''
    tokens = []
    lines_per_block = int(lines_per_block)
    for i in range(lines_per_block):
        tokens.append(block_file.readline()[:-1])
    
    return tokens,block_file.tell()



def find_next_block(block_file):
    '''
    Encuentra el siguiente bloque grande
    '''
    lines_per_block,small_blocks = block_file.readline().split(' ')[:2]
    while int(small_blocks) == -1:
        for i in range(int(lines_per_block)):
            nada =block_file.readline()
            
        lines_per_block,small_blocks = block_file.readline().split(' ')[:2]

    return lines_per_block,small_blocks

def write_new_block(new_file,tokens,mode):
    write_pos = -1
    if mode:
        new_file.write(str(len(tokens)) + ' ')
        write_pos = new_file.tell()
        for i in range(1000):
            new_file.write(' ')
        new_file.write('\n')
    else:
        new_file.write(str(len(tokens)) + " -1\n")
  

    for tok in tokens:
        new_file.write(tok + '\n')

    return write_pos




def merge(archivo, big_blocks):
    if big_blocks == 1:
        os.rename(archivo,'sorted_tokens.txt')
        return ''
    with open(archivo,'r') as blocks, open('new_file.txt','w+') as new_file:
        new_big_blocks = 0
        for i in range(1,big_blocks,2):
            if i == big_blocks:
                #se escribe el ultimo bloque
                lines_per_block,small_blocks = blocks.readline().split(" ")[:2]
                tokens_bloque, pos_bloque = get_next_small_block(blocks,lines_per_block)

                write_position = -1
                output_tokens = []
                total_small_blocks = 0 
                bytes_written = 0


                p = 0
                while True:
                    if p == len(tokens_bloque):
                        if small_blocks == 0:
                            break
                        blocks.seek(pos_bloque)
                        lines_per_block,_ = blocks.readline().split(" ")[:2]
                        tokens_bloque, pos_bloque = get_next_small_block(blocks,lines_per_block)
                        small_blocks -= 1
                        p = 0
                    if bytes_written + len(tokens_bloque1[p]) > BLOCKSIZE:
                        if total_small_blocks == 0:
                            write_position = write_new_block(new_file,output_tokens,1)
                        else:
                            write_new_block(new_file,output_tokens,0)
                        total_small_blocks += 1
                        output_tokens = []
                        bytes_written = 0
                    output_tokens.append(tokens_bloque1[p])
                    p += 1
                if len(output_tokens) > 0:
                    if total_small_blocks == 0:
                        write_position = write_new_block(new_file,output_tokens,1)
                    else:
                        write_new_block(new_file,output_tokens,0)
                    total_small_blocks += 1
                new_file.seek(write_position)
                new_file.write(str(total_small_blocks))
                new_big_blocks += 1
                

            else:
                
        
                lines_per_block1,small_blocks_1 = blocks.readline().split(" ")[:2]



                small_blocks_1 = int(small_blocks_1) - 1

                tokens_bloque1, pos_bloque1 = get_next_small_block(blocks,lines_per_block1)
                

                lines_per_block2,small_blocks_2 =  find_next_block(blocks)
                small_blocks_2 = int(small_blocks_2) - 1


                tokens_bloque2, pos_bloque2 = get_next_small_block(blocks,lines_per_block2)


                write_position = -1
                output_tokens = []
                total_small_blocks = 0 
                p1 = 0
                p2 = 0
                bytes_written = 0

                while True:
                    if p1 == len(tokens_bloque1):
                        if small_blocks_1 == 0:
                            break
                        blocks.seek(pos_bloque1)
                        lines_per_block1, _ = blocks.readline().split(" ")[:2]
                        tokens_bloque1, pos_bloque1 = get_next_small_block(blocks,lines_per_block1)
                        small_blocks_1 -= 1
                        p1 = 0 
                        
                    if p2 == len(tokens_bloque2):
                        if small_blocks_2 == 0:
                            break
                        blocks.seek(pos_bloque2)
                        lines_per_block2, _ = blocks.readline().split(" ")[:2]

                        tokens_bloque2, pos_bloque2 = get_next_small_block(blocks,lines_per_block2)
                        small_blocks_2 -= 1
                        p2 = 0

                    option_1 = bytes_written + len(tokens_bloque1[p1]) + 1
                    option_2 = bytes_written + len(tokens_bloque2[p2]) + 1

                    if (tokens_bloque1[p1] <= tokens_bloque2[p2] and option_1 > BLOCKSIZE) or (tokens_bloque1[p1] > tokens_bloque2[p2] and option_2 > BLOCKSIZE): 
                        if total_small_blocks == 0:
                            write_position = write_new_block(new_file,output_tokens,1)
                        else:
                            write_new_block(new_file,output_tokens,0)
                        total_small_blocks += 1
                        output_tokens = []
                        bytes_written = 0
                        option_1 = len(tokens_bloque1[p1]) 
                        option_2 = len(tokens_bloque2[p2])
                    
                    if tokens_bloque1[p1] == tokens_bloque2[p2]:
                        bytes_written = option_1 
                        output_tokens.append(tokens_bloque1[p1]) 
                        p1 = p1 + 1
                        p2 = p2 + 1
                    elif tokens_bloque1[p1] < tokens_bloque2[p2]:
                        bytes_written = option_1
                        output_tokens.append(tokens_bloque1[p1])
                        p1 = p1 + 1
                    else:
                        bytes_written = option_2
                        output_tokens.append(tokens_bloque2[p2])
                        p2 = p2 + 1
                    
                while True:
                    if p1 == len(tokens_bloque1):
                        if small_blocks_1 == 0:
                            break
                        blocks.seek(pos_bloque1)
                        lines_per_block1,_ = blocks.readline().split(" ")[:2]
                        tokens_bloque1, pos_bloque1 = get_next_small_block(blocks,lines_per_block1)
                        small_blocks_1 -= 1
                        p1 = 0
                    if bytes_written + len(tokens_bloque1[p1]) > BLOCKSIZE:
                        if total_small_blocks == 0:
                            write_position = write_new_block(new_file,output_tokens,1)
                        else:
                            write_new_block(new_file,output_tokens,0)
                        total_small_blocks += 1
                        output_tokens = []
                        bytes_written = 0
                    output_tokens.append(tokens_bloque1[p1])
                    p1 += 1

                while True:
                    if p2 == len(tokens_bloque2):
                        if small_blocks_2 == 0:
                            break
                        blocks.seek(pos_bloque2)
                        lines_per_block2,_ = blocks.readline().split(" ")[:2]
                        tokens_bloque2, pos_bloque2 = get_next_small_block(blocks,lines_per_block2)
                        small_blocks_2 -= 1
                        p2 = 0
                    if bytes_written + len(tokens_bloque2[p2]) > BLOCKSIZE:
                        if total_small_blocks == 0:
                            write_position = write_new_block(new_file,output_tokens,1)
                        else:
                            write_new_block(new_file,output_tokens,0)

                        total_small_blocks += 1
                        output_tokens = []
                        bytes_written = 0
                    output_tokens.append(tokens_bloque2[p2])
                    p2 += 1

                if len(output_tokens) > 0:
                    if total_small_blocks == 0:
                        write_position = write_new_block(new_file,output_tokens,1)
                    else:
                        write_new_block(new_file,output_tokens,0)
                    total_small_blocks += 1

                new_file.seek(write_position)
                new_file.write(str(total_small_blocks))
                new_file.seek(0,io.SEEK_END)
                new_big_blocks += 1

                blocks.seek(pos_bloque2)

    os.remove(archivo)
    os.rename('new_file.txt', archivo)
    merge(archivo, new_big_blocks)



def create_tfidf(sorted_tokens_file_name,n_tweets):
    with open(sorted_tokens_file_name,'r') as sorted_tokens:
        lines_per_block,small_blocks = sorted_tokens.readline().split(" ")[:2]
        tokens_in_block,pos_bloque = get_next_small_block(sorted_tokens,lines_per_block)
        small_blocks = int(small_blocks) - 1

        while small_blocks >= 0:
            print(small_blocks)
            for tok in tokens_in_block:
                with open('collections/list_' + convert_key_to_file_name(tok) + '.txt','r') as collection:
                    df = int(collection.readline()[:-1])
                    idf = np.log10(n_tweets/df)
                    
                    for tweet_id in range(1,n_tweets+1):#para cada documento poner el tf-idf de la palabra tok
                        path_vector = 'vectors/tweet_id_' + str(tweet_id).zfill(6)
                        Path(path_vector).touch(exist_ok=True)
                        norma = 0
                        with open(path_vector,'r+') as vector_tweet:
                            vector_tweet.seek(0,io.SEEK_END)
                            if vector_tweet.tell() == 0:
                                for i in range(1000):
                                    vector_tweet.write(' ')
                                vector_tweet.write('\n')
                            else:
                                vector_tweet.seek(0)
                                norma = float(vector_tweet.readline().split(' ')[0])
                                vector_tweet.seek(0,io.SEEK_END)
                            file_pos = collection.tell()
                            line = collection.readline()
                            found = 0



                            if line != '':# pushear tf-idf al final del vector para el tweet 
                                if line[-1] == '\n':
                                    line = line[:-1]
                                tweet_name,tf = line.split('-')
                                tf_idf = np.log10(1 + int(tf))*idf
                                
                                if tweet_name == 'tweet_' + str(tweet_id).zfill(6):
                                    vector_tweet.write(str(tf_idf) + '\n')
                                    norma += tf_idf*tf_idf
                                    found = 1

                            if not found: 
                                vector_tweet.write('0\n')
                                collection.seek(file_pos)
                            vector_tweet.seek(0)
                            vector_tweet.write(str(norma))
            l = sorted_tokens.readline()
            if l == '':
                break
            lines_per_block, _ = l.split(' ')[:2]
            tokens_in_block, pos_bloque = get_next_small_block(sorted_tokens, lines_per_block)
            small_blocks -= 1



def normalize(n_tweets):
    for tweet_id in range(1, n_tweets + 1):
        vector_file_name = 'vectors/tweet_id_' + str(tweet_id).zfill(6)
        norm_file_name = 'norm/tweet_id_' + str(tweet_id).zfill(6)
        with open(vector_file_name) as vector, open(norm_file_name, 'a') as norm_file:
            norm = vector.readline()
            while True:
                line = vector.readline()
                if line == '':
                    break
                tf_idf = line
                try:
                    tf_idf = float(tf_idf)/np.sqrt(float(norm))
                except:
                    return None
                norm_file.write(str(tf_idf) + '\n')



def create_tf_query(query_tokens):
    query_tf = {}
    for tok in query_tokens:
        if query_tf.get(tok):
            query_tf[tok] += 1
        else:
            query_tf[tok] = 1 
    
    return (sorted(query_tf.keys()),query_tf)



def create_unit_vector_query(sorted_tokens_file_name,n_tweets,sorted_keys_query,tf_dictionary):
    norm = 0
    with open(sorted_tokens_file_name,'r') as sorted_tokens:
        lines_per_block,small_blocks = sorted_tokens.readline().split(" ")[:2]
        tokens_in_block,pos_bloque = get_next_small_block(sorted_tokens,lines_per_block)
        small_blocks = int(small_blocks) - 1
        p = 0
        finish = 0
   


        while small_blocks >= 0:
            for tok in tokens_in_block:
                idf = -1
                with open('collections/list_' + convert_key_to_file_name(tok) + '.txt','r') as collection:
                    df = int(collection.readline()[:-1])
                    idf = np.log10(n_tweets/df)
            
                while p < len(sorted_keys_query) and tok > sorted_keys_query[p]:
                    p = p + 1

                if p == len(sorted_keys_query):
                    finish = 1
                    break

                if tok == sorted_keys_query[p]:
                    tf = tf_dictionary[sorted_keys_query[p]]
                    tf_dictionary[sorted_keys_query[p]] = np.log10(1 + int(tf))*idf

                    norm += tf_dictionary[sorted_keys_query[p]]*tf_dictionary[sorted_keys_query[p]]
                    p = p+1

                
                if p == len(sorted_keys_query):
                    finish = 1
                    break
                
                
            l = sorted_tokens.readline()
            if l == '' or finish:
                break
            lines_per_block, _ = l.split(' ')[:2]
            tokens_in_block, pos_bloque = get_next_small_block(sorted_tokens, lines_per_block)
            small_blocks -= 1    

    with open(sorted_tokens_file_name,'r') as sorted_tokens,open('query_unit_vector.txt','w') as query_file:
        lines_per_block,small_blocks = sorted_tokens.readline().split(" ")[:2]
        tokens_in_block,pos_bloque = get_next_small_block(sorted_tokens,lines_per_block)
        small_blocks = int(small_blocks) - 1
        p = 0
        finish = 0

        while small_blocks >= 0:
            for tok in tokens_in_block:
                
                while  p < len(sorted_keys_query) and tok > sorted_keys_query[p]:
                    p = p + 1

                if p < len(sorted_keys_query) and tok == sorted_keys_query[p] :
                    query_file.write(str(tf_dictionary[sorted_keys_query[p]]/np.sqrt(norm)) + '\n')
                    p = p + 1
                else:
                    query_file.write(str(0.0) + '\n')

            l = sorted_tokens.readline()
            if l == '' or finish:
                break
            lines_per_block, _ = l.split(' ')[:2]
            tokens_in_block, pos_bloque = get_next_small_block(sorted_tokens, lines_per_block)
            small_blocks -= 1




def create_tf_query(query_tokens):
    query_tf = {}
    for tok in query_tokens:
        if query_tf.get(tok):
            query_tf[tok] += 1
        else:
            query_tf[tok] = 1 
    
    return (sorted(query_tf.keys()),query_tf)




def dotproduct(tweet, query):
    score = 0
    query_val = query.readline()
    tweet_val = tweet.readline()
    while query_val != '':
        score += float(tweet_val) * float(query_val)
        query_val = query.readline()
        tweet_val = tweet.readline()
    return score


def store_tweets():
    n_id_tweet = 1
    df = pd.read_csv('ds0.csv')
    for i in range(1, 2506):
        with open('tweets/tweet_id_' + str(i).zfill(6) + '.txt', 'a') as tweet_file:
            tweet_file.write(str(df.iloc[i-1,0]))

# store_tweets()


def topk(n_id_tweet, K):
    h = PriorityQueue()

    for tweet in range(1, n_id_tweet - 1):
        tweet_vector = 'norm/tweet_id_' + str(tweet).zfill(6)
        query_vector = 'query_unit_vector.txt'
        with open(tweet_vector) as curr_tweet, open(query_vector) as query,open('tweets/tweet_id_' + str(tweet).zfill(6) + '.txt','r') as tweet_content:
            content = tweet_content.readline()
            score = dotproduct(curr_tweet, query)
            if len(h.queue) < K:
                h.put((score,tweet,content))
            else :
                top = h.queue[0]
                if score > top[0]:
                    h.get()
                    h.put((score,tweet, content))
    

    result = []
    while not h.empty():
        temp = h.get()
        if temp[0] != 0:
            result = [temp] + result

    return result


# postings_list = 0
# #tamanio total 311206
# n_id_tweet = 1
# big_blocks = 0 
# for n_page in range(0,2500,14):
    
#     print(n_page)
#     df = pd.read_csv('ds0.csv',skiprows=n_page,nrows=14)
#     block_dictionary = {}
#     for k in range(df.shape[0]):
#         token_stream = procesamiento(str(df.iloc[k,0]))
#         str_tweet_id = 'tweet_' + str(n_id_tweet).zfill(6)
#         n_id_tweet += 1 
#         block_dictionary = spimi(token_stream, str_tweet_id, block_dictionary)
#     write_to_memory(sorted(block_dictionary.keys()),block_dictionary)
#     big_blocks += 1

# print(n_id_tweet)
# merge('temp.txt',big_blocks)


# create_tfidf('sorted_tokens.txt',n_id_tweet-1)



# normalize(n_id_tweet-1)




# x, y = create_tf_query(procesamiento('coronavirus in latin america'))



# create_unit_vector_query('sorted_tokens.txt', n_id_tweet - 1, x, y)