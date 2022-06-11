
create database proyecto2_base_de_datos_2;
\c proyecto2_base_de_datos_2

create table tweets(id serial,texto text, a double precision, subjectivity double precision);

copy tweets(texto,a,subjectivity) from '{any dir}/ds0.csv' delimiter ',' csv header;

ALTER TABLE tweets ADD COLUMN weighted_tsv tsvector;

UPDATE tweets SET  
    weighted_tsv = x.weighted_tsv
FROM (  
    SELECT id,
           setweight(to_tsvector('english', COALESCE(texto,'')), 'A') 
           AS weighted_tsv
     FROM tweets
) AS x
WHERE x.id = tweets.id;

CREATE INDEX weighted_tsv_idx ON tweets USING GIN (weighted_tsv);
