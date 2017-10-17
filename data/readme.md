FOLDERS:

ml - machine learning datasets
solr - a solr index template. To start a solr server running at the background (which is needed before running the python script to fetch data), cd into your downloaded Solr executables. Type

./bin/solr -s [..../chase/data/solr]

i.e., -s with path to the 'solr' dir in this folder.

This will run a solr instance at the background on the default port 8983. You can see the server by typing the url into your browser: ''

To stop that server, open a console run

bin/solr stop -p 8983


to commit changes to index:
http://localhost:8983/solr/chase/update?commit=true
