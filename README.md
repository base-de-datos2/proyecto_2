# proyecto_2

|Nombre|Participacion|Nota
|-|-|-|
Francisco Magot|Creacion del indice en memoria secundaria, conexion con postgres, topk y servicio web|-|
Eric Bracamonte|Creacion del indice en memoria secundaria, conexion con postgres,topk y servicio web|-|


> Video de demostración

https://youtu.be/9dS-W3B6TUo

> Levantar el proyecto

Se tiene que estar en un sistema operativo POSIX (LINUX o MAC) y, con un usuario postgres, correr el script *database.sql* (tambien se 
tendria que cambiar password en el connect de *app.py*). Luego, ejecutar *init_index.py*. Finalmente, prender el servidor de flask con el 
commando *python app.py*.


> Dominio de Datos

Este motor de búsqueda funciona con una colección de tweets conseguidos del siguiente enlace: https://www.kaggle.com/harshrey/tweets-covid-sentimentvalues. Todos son tweets relacionados al COVID-19. Consiste de 3 columnas de las cuales solo se utilizará la que contiene el contenido del tweet. Un tweet por naturaleza tiene un límite de 280 caracteres por tweet.

> Construcción del indice invertido

Para construir el indice invertido, hemos utilizado el método de SPIMI (Single pass in memory indexing). Para esto, hemos calculado cuantos tweets entrarían en una página de memoria en el peor caso (este siendo que todos los 280 caracteres haya sido utilizados) y esto sería 14 tweets por página (piso de: 4096B en una página / 280 caracteres por tweet = 14 tweets por página). Este número se utilizará para procesar los tweets en bloques de 14 por cada iteración del SPIMI. 

Para un bloque de tweets se hace el siguiente procedimiento:

1. Tokenizar el contenido de todos los tweets y sacarle la raíz a cada uno de los tokens
2. Una vez tokenizado el contenido, se debe crear el índice invertido del bloque. La colleción de tweets done aparece un token será guardado en un archivo llamado "list_(token).txt". Cabe recalcar que si un token se repite en otro bloque de tweets, los tweets del nuevo bloque se agregarán a la lista existente del token, así ahorrando el procedimiento de juntarlos en el paso del merge.
3. Una vez calculado el índice invertido del bloque, cada token se guardará en un archivo ordenado alfabéticamente (solo por bloque).

Este procedimiento se ejecutará hasta que no hayan más tweets por procesar. Una vez terminado el proceso, se procedará a llamar a la función "merge". Esta función tiene el propósito de cargar bloque por bloque del archivo de tokens y juntarlos en orden alfabético.

> Similitud de coseno

Una vez completado el proceso del merge, para cada uno de los documentos, se calculará un vector unitario en base a la ocurrencia y la rareza de los términos que contiene (peso tf-idf). Esto es calculado para luego poder hacer un producto punto entre dos vectores distintos y comparar su similitud.

> Construccion del indice en postgres

Se creó una tabla tweets que tuviera como un id y los mismos atributos de ds0.csv. Después de insertar los datos se le agrego una nueva columna,
weighted_tsv, que sería un tsvector. Finalmente, se le aplico un índice GIN a dicho vector.  