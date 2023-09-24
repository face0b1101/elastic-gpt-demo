import openai
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from elasticsearch import Elasticsearch
from yaml.loader import SafeLoader

from elastic_gpt_demo.settings.settings import (
    APP_NAME,
    CORPUS_DESCRIPTION,
    DEMO_PW,
    DEMO_USER,
    ES_CLOUD_ID,
    ES_CLOUD_PW,
    ES_CLOUD_USER,
    ES_INDEX,
    ES_SEARCH_FIELD,
    GENAI_MODEL,
    IMAGE_URL,
    OPENAI_API_KEY,
    QUERY_IMPROVEMENT,
    SANITY_CHECK,
)

# This code is part of an Elastic Blog showing how to combine
# Elasticsearch's search relevancy power with
# OpenAI's GPT's Question Answering power
# https://www.elastic.co/blog/chatgpt-elasticsearch-openai-meets-private-data

# Code is presented for demo purposes but should not be used in production
# You may encounter exceptions which are not handled in the code

with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)


# Required Environment Variables
# openai_api - OpenAI API Key
# cloud_id - Elastic Cloud Deployment ID
# cloud_user - Elasticsearch Cluster User
# cloud_pass - Elasticsearch User Password

openai.api_key = OPENAI_API_KEY

hashed_pass = stauth.Hasher([DEMO_PW]).generate()[0]
creds = config["credentials"]
creds["usernames"] = {}
creds["usernames"][DEMO_USER] = {"password": hashed_pass, "name": DEMO_USER}

authenticator = stauth.Authenticate(
    creds,
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)


def es_connect(cloud_id: str, cloud_user: str, cloud_pw: str) -> Elasticsearch:
    """
    Connects to Elasticsearch using the provided cloud_id, cloud_user, and cloud_pw.

    Args:
        cloud_id (str): The Cloud ID of the Elasticsearch cluster.
        cloud_user (str): The username for the Elasticsearch cluster.
        cloud_pw (str): The password for the Elasticsearch cluster.

    Returns:
        Elasticsearch: An instance of the Elasticsearch client connected to the cluster.
    """
    return Elasticsearch(cloud_id=cloud_id, http_auth=(cloud_user, cloud_pw))


def search(query_text: str) -> tuple:
    """
    Search for documents in Elasticsearch based on a query text.

    Args:
        query_text (str): The text to search for.

    Returns:
        tuple: A tuple containing the body content of the top search result,
               the URL of the top search result, and a list of URLs of the top search results.
    """
    # Connect to Elasticsearch
    es = es_connect(
        cloud_id=ES_CLOUD_ID, cloud_user=ES_CLOUD_USER, cloud_pw=ES_CLOUD_PW
    )

    # Define Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{"match": {"title": {"query": query_text, "boost": 1}}}],
            "filter": [{"exists": {"field": ES_SEARCH_FIELD}}],
        }
    }

    knn = {
        "field": ES_SEARCH_FIELD,
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text,
            }
        },
        "boost": 24,
    }

    fields = ["title", "body_content", "url"]

    # Execute the search query
    resp = es.search(
        index=ES_INDEX, query=query, knn=knn, fields=fields, size=5, source=False
    )

    # Get the body content and URL of the top search result
    body = resp["hits"]["hits"][0]["fields"]["body_content"][0]
    url = resp["hits"]["hits"][0]["fields"]["url"][0]

    # Get the URLs of the top search results
    urls = [
        resp["hits"]["hits"][i]["fields"]["url"][0]
        for i in range(0, max(4, len(resp["hits"]["hits"])))
    ]

    return body, url, urls


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncates the given text to a specified number of tokens.

    Args:
        text: The input text to truncate.
        max_tokens: The maximum number of tokens to keep in the truncated text.

    Returns:
        The truncated text.

    """
    # Split the text into tokens
    tokens = text.split()

    # Check if the number of tokens is less than or equal to the maximum allowed
    if len(tokens) <= max_tokens:
        return text

    # Join the first max_tokens tokens and return the truncated text
    return " ".join(tokens[:max_tokens])


def sanity_check_gpt(query: str, description: str, model: str) -> bool:
    """
    Perform a sanity check using the GPT model to determine if a set of documents might help answer a given question.

    Args:
        query (str): The question/query to check against the documents.
        description (str): The description of the documents.
        model (str): The name of the GPT model to use.

    Returns:
        bool: True if the answer is "Yes", False otherwise.
    """
    # Generate the prompt
    prompt = f'only reply with "yes" or "no". There are a set of documents with a description of "{description}". Do you think they might help answer the question "{query}"?'

    # Make the API call to GPT
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    # Print the prompt and response for debugging purposes
    print(
        "Prompt=",
        prompt,
        "\n",
        "response=",
        response["choices"][0]["message"]["content"],
    )

    # Extract the answer from the response
    answer = response["choices"][0]["message"]["content"]
    answer = answer.strip(".")

    # Return True if the answer is "Yes", False otherwise
    return answer == "Yes"


def query_improvement_gpt(query: str, description: str, model: str) -> str:
    """
    Generate a prompt to ask the user for a search query based on a given description.

    Args:
        query (str): The question being asked.
        description (str): The description of the set of documents.
        model (str): The OpenAI model to use for generating the response.

    Returns:
        str: The generated search query.
    """
    # Generate the prompt message
    prompt = (
        f'There are a set of documents with a description of "{description}". '
        f'I\'m trying to answer this question "{query}" what would you search in a search engine '
        f"to get back the best document to answer this question? Please only reply with the search string"
    )

    # Create the chat completion request
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    # Extract the answer from the response
    answer = response["choices"][0]["message"]["content"]

    # Print the prompt and response for debugging purposes
    print("Prompt=", prompt, "\n", "response=", answer)

    return answer


def chat_gpt(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 1024,
    max_context_tokens: int = 4000,
    safety_margin: int = 5,
) -> str:
    """
    Generate a chat response using a GenAI Model (default: OpenAI GPT-3.5 Turbo model)

    Args:
        prompt (str): The user's prompt.
        model (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.
        max_context_tokens (int, optional): The maximum number of tokens in the model's context. Defaults to 4000.
        safety_margin (int, optional): The safety margin for truncating the prompt. Defaults to 5.

    Returns:
        str: The generated response.
    """
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(
        prompt, max_context_tokens - max_tokens - safety_margin
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": truncated_prompt},
        ],
    )

    return response["choices"][0]["message"]["content"]


def main():
    """
    Generates the main user interface for the chat application.

    This function sets up the app title and image, creates the main chat form,
    adds checkboxes for sanity check and query improvement options, gets the user input query,
    and generates and displays the response on form submission.

    Parameters:
        None

    Returns:
        None
    """
    # Set up the app title and image
    st.title(APP_NAME)
    st.image(IMAGE_URL)

    # Main chat form
    with st.form("chat_form"):
        # Add checkboxes for sanity check and query improvement options
        SANITY_CHECK = st.checkbox(label="Sanity check query", value=SANITY_CHECK)
        QUERY_IMPROVEMENT = st.checkbox(
            label="Try to improve query", value=QUERY_IMPROVEMENT
        )

        # Get user input query
        query = st.text_input("You: ")
        submit_button = st.form_submit_button("Send")

    # Generate and display response on form submission
    negResponse = "I'm unable to answer the question based on the information I have from the top document."
    if submit_button:
        original_query = query
        sensible = True

        # Perform sanity check if option is selected
        if SANITY_CHECK:
            sensible = sanity_check_gpt(query, CORPUS_DESCRIPTION, GENAI_MODEL)
            if not sensible:
                st.write(
                    f'ChatGPT: I\'m not sure If I can help, the documents I have seem to be on a different topic. They are described like this "{CORPUS_DESCRIPTION}"'
                )
            else:
                st.write(
                    "ChatGPT: You are in luck, I have some documents that might be able to help answer that"
                )

        if sensible:
            # Improve query if option is selected
            if QUERY_IMPROVEMENT:
                query = query_improvement_gpt(query, CORPUS_DESCRIPTION, GENAI_MODEL)
                query = query.strip('-"')
                st.write(
                    f"ChatGPT: To find the best document I will search for the following:\n{query}"
                )

            # Perform search based on query
            resp, url, urls = search(query)

            # Set up prompt for chat GPT
            prompt = f"Answer this question: {original_query}\nUsing only the information from this document: {resp}\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"

            # Get answer from chat GPT
            answer = chat_gpt(prompt)

            # Display answer or negResponse if answer is not found
            if negResponse in answer:
                st.write(f"ChatGPT: {answer.strip()}")
            else:
                st.write(f"ChatGPT: {answer.strip()}")

            # Display top 5 search results
            st.write("Top 5 results were:\n\n\n" + "\n\n".join(urls))


name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    authenticator.logout("Logout", "main")
    st.write(f"Welcome *{name}*")
    main()

elif authentication_status is False:
    st.error("Username/password is incorrect")

elif authentication_status is None:
    st.warning("Please enter your username and password")
