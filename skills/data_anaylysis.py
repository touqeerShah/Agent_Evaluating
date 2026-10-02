from openai import OpenAI
import pandas as pd
import json
import duckdb
from pydantic import BaseModel, Field
from IPython.display import Markdown

from helper import get_openai_api_key
from skills.lookuo_sales_data import lookup_sales_data
import phoenix as px
import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from openinference.instrumentation import TracerProvider
from helper import get_openai_api_key, get_phoenix_endpoint

# initialize the OpenAI client
openai_api_key = get_openai_api_key()
client = OpenAI(api_key=openai_api_key)
PROJECT_NAME = "tracing-agent"
tracer_provider = register(
    project_name=PROJECT_NAME, endpoint=get_phoenix_endpoint() + "v1/traces"
)
MODEL = "gpt-4o-mini"
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)
MODEL = "gpt-4o-mini"

# Construct prompt based on analysis type and data subset
from http import client

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""


# code for tool 2
@tracer.tool()
def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis"""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    analysis = response.choices[0].message.content
    return analysis if analysis else "No analysis could be generated"


example_data = lookup_sales_data(
    "Show me all the sales for store 1320 on November 1st, 2021"
)
print(example_data)
print(
    analyze_sales_data(prompt="what trends do you see in this data", data=example_data)
)
