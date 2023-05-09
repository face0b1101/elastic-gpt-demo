import os
import streamlit as st
import streamlit_authenticator as stauth
import openai
from elasticsearch import Elasticsearch
import yaml

# This code is part of an Elastic Blog showing how to combine
# Elasticsearch's search relevancy power with 
# OpenAI's GPT's Question Answering power
# https://www.elastic.co/blog/chatgpt-elasticsearch-openai-meets-private-data

# Code is presented for demo purposes but should not be used in production
# You may encounter exceptions which are not handled in the code

from yaml.loader import SafeLoader
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


# Required Environment Variables
# openai_api - OpenAI API Key
# cloud_id - Elastic Cloud Deployment ID
# cloud_user - Elasticsearch Cluster User
# cloud_pass - Elasticsearch User Password

openai.api_key = os.environ['openai_api']
index = os.getenv('index','search-elastic-docs')
model = os.getenv('model','gpt-3.5-turbo-0301')
field = os.getenv('field','title-vector')
app_name = os.getenv('app_name','ElasticDocs GPT')
corpus_description = os.getenv('corpus_description','documents about every subject')
image_url = os.getenv('image_url','documents about every subject')
sanity_check = bool(os.getenv('app_name','true'))
query_improvement = bool(os.getenv('query_improvement','true'))
demo_username = os.getenv('demo_username','demo-user')
demo_password = os.getenv('demo_password','')

hashed_pass=stauth.Hasher([demo_password]).generate()[0]
creds=config['credentials']
creds['usernames']={}
creds['usernames'][demo_username] = {'password': hashed_pass, 'name': demo_username}

authenticator = stauth.Authenticate(
    creds,
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized'])

# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    return es

# Search ElasticSearch index and return body and URL of the result
def search(query_text):
    cid = os.environ['cloud_id']
    cp = os.environ['cloud_pass']
    cu = os.environ['cloud_user']
    es = es_connect(cid, cu, cp)

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": field
                }
            }]
        }
    }

    knn = {
        "field": field,
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "body_content", "url"]
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=5,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]
    urls=[]

    for i in range(0,max(4,len(resp["hits"]["hits"]))):
        urls.append(resp['hits']['hits'][i]['fields']['url'][0])

    return body, url,urls

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

def sanity_check_gpt(query,description,model):
    prompt=f"only reply with \"yes\" or \"no\". There are a set of documents with a description of \"{description}\". Do you think they might help answer the question \"{query}\"?"
    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}])
    print("Prompt=",prompt,"\n","response=",response["choices"][0]["message"]["content"] )
    answer = response["choices"][0]["message"]["content"]
    answer =answer.strip(".")
    return answer == "Yes"

def query_improvement_gpt(query,description,model):
    prompt=f"There are a set of documents with a description of \"{description}\". I'm trying to answer this question \"{query}\" what would you search in a search engine to get back the best document to answer this question? Please only reply with the search string"
    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}])

    answer = response["choices"][0]["message"]["content"]
    print("Prompt=",prompt,"\n","response=",answer)
    return answer

# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, model="gpt-3.5-turbo", max_tokens=1024, max_context_tokens=4000, safety_margin=5):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": truncated_prompt}])

    return response["choices"][0]["message"]["content"]



def app_main():
    st.title(app_name)
    st.image(image_url)
    # Main chat form
    with st.form("chat_form"):
        global sanity_check,query_improvement
        sanity_check = st.checkbox(label='Sanity check query',value= sanity_check)
        query_improvement = st.checkbox(label='Try to improve query',value=query_improvement)
        query = st.text_input("You: ")
        submit_button = st.form_submit_button("Send")

    # Generate and display response on form submission
    negResponse = "I'm unable to answer the question based on the information I have from the top document."
    if submit_button:
        original_query= query
        sensible=True
        if sanity_check:
            sensible = sanity_check_gpt(query,corpus_description,model)
            if not sensible:
                st.write(f"ChatGPT: I'm not sure If I can help, the documents I have seem to be on a different topic. They are described like this \"{corpus_description}\"")
            else:
                st.write(f"ChatGPT: You are in luck, I have some documents that might be able to help answer that")
            
        if sensible:
            if query_improvement:
                query = query_improvement_gpt(query,corpus_description,model)
                query=query.strip("-\"")
                st.write(f"ChatGPT: To find the best document I will serch for the following:\n{query}")
            resp, url,urls = search(query)
            prompt = f"Answer this question: {original_query}\nUsing only the information from this document: {resp}\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"
            answer = chat_gpt(prompt)
            
            if negResponse in answer:
                st.write(f"ChatGPT: {answer.strip()}")
            else:
                st.write(f"ChatGPT: {answer.strip()}")
            st.write("Top 5 results were:\n\n\n"+"\n\n".join(urls))

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
    app_main()
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')


