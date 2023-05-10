# Elastic Chat GPT Demo
Combining the search power of Elasticsearch with the Question Answering power of GPT

Originally based on and forked from this blog
[Blog - ChatGPT and Elasticsearch: OpenAI meets private data](https://www.elastic.co/blog/chatgpt-elasticsearch-openai-meets-private-data)


![diagram](https://raw.githubusercontent.com/jeffvestal/ElasticDocs_GPT/main/images/ElasticChat%20GPT%20Diagram%20-%20No%20line%20text.jpeg)

1. Python interface accepts user questions
- Generate a hybrid search request for Elasticsearch
- BM25 match on the title field
- kNN search on the title-vector field
- Boost kNN search results to align scores
- Set size=1 to return only the top scored document
2. Search request is sent to Elasticsearch
3. Documentation body and original url are returned to python
4. API call is made to OpenAI ChatCompletion
- Prompt: "answer this question <question> using only this document <body_content from top search result>"
5. Generated response is returned to python
6. Python adds on original documentation source url to generated response and prints it to the screen for the user

  To make this work for your data. Please set the appropriate environmental variables either through a .env file or through k8s.
More examples on how to do this will follow.
  
  You can run this standalone, in a docker container or in kubernetes

## To run in Docker:
 
 docker build -t elastic-chatgpt .

 cp ./example.env .env

 If you are on Mac cmd+shift+. in finder to view hidden files
 Edit the env file to add your secrets

 docker run -p --env-file .env 8501:8501 elastic-chatgpt 
