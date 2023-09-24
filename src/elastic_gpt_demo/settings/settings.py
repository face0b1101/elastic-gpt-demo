#!/usr/bin/env python3

from decouple import config
from unipath import Path

BASE_DIR = Path(__file__).parent

LOG_LEVEL = config("LOG_LEVEL", cast=str, default="WARNING")
DEFAULT_TZ = config("TZ", cast=str, default="Europe/London")

ES_CLOUD_ID = config("ES_CLOUD_ID", cast=str)
ES_CLOUD_USER = config("ES_CLOUD_USER", cast=str, default="elastic")
ES_CLOUD_PW = config("ES_CLOUD_PW", cast=str, default="")
ES_INDEX = config("ES_INDEX", cast=str, default="search-elastic-docs")
ES_SEARCH_FIELD = config("ES_SEARCH_FIELD", cast=str, default="text")

OPENAI_API_KEY = config("OPENAI_API_KEY", cast=str, default=None)
GENAI_MODEL = config("GENAI_MODEL", cast=str, default="gpt-3.5-turbo-0301")

APP_NAME = config("APP_NAME", cast=str, default="Elasticsearch GPT")
IMAGE_URL = config("IMAGE_URL", cast=str, default="")
CORPUS_DESCRIPTION = config("CORPUS_DESCRIPTION", cast=str, default="UK Legislation")

DEMO_USERNAME = config("DEMO_USERNAME", cast=str, default="demo-user")
DEMO_PW = config("DEMO_PASSWORD", cast=str, default="")

SANITY_CHECK = config("SANITY_CHECK", cast=bool, default=False)
QUERY_IMPROVEMENT = config("QUERY_IMPROVEMENT", cast=bool, default=True)
