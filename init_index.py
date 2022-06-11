from creacion_del_indice import * 
postings_list = 0
#tamanio total 311206
n_id_tweet = 1
big_blocks = 0 
for n_page in range(0,2500,14):
    
    print(n_page)
    df = pd.read_csv('ds0.csv',skiprows=n_page,nrows=14)
    block_dictionary = {}
    for k in range(df.shape[0]):
        token_stream = procesamiento(str(df.iloc[k,0]))
        str_tweet_id = 'tweet_' + str(n_id_tweet).zfill(6)
        n_id_tweet += 1 
        block_dictionary = spimi(token_stream, str_tweet_id, block_dictionary)
    write_to_memory(sorted(block_dictionary.keys()),block_dictionary)
    big_blocks += 1

print(n_id_tweet)
merge('temp.txt',big_blocks)


create_tfidf('sorted_tokens.txt',n_id_tweet-1)



normalize(n_id_tweet-1)




store_tweets()
