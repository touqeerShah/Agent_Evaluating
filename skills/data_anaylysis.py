
# Construct prompt based on analysis type and data subset
from http import client

import os
from openai import OpenAI
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from skills.lookuo_sales_data import lookup_sales_data


# Local Ollama
MODEL = "qwen2.5:7b-instruct"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=180.0,
)

# Local Phoenix
PROJECT_NAME = "tracing-agent"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint="http://localhost:6006/v1/traces",
    protocol="http/protobuf",
    batch=False,
    sampler=ALWAYS_ON,
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)


DATA_ANALYSIS_PROMPT = """
Analyze the following data:
{data}

Answer the following question:
{prompt}

Base your answer only on the supplied data.
Explain any limitations that affect your conclusions.
"""


@tracer.tool()
def analyze_sales_data(prompt: str, data: str) -> str:
    """Analyze sales data using the local Ollama model."""

    if not data.strip():
        raise ValueError("No sales data was provided.")

    if data.startswith("Error accessing data:"):
        raise ValueError(data)

    formatted_prompt = DATA_ANALYSIS_PROMPT.format(
        data=data,
        prompt=prompt,
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "user", "content": formatted_prompt},
        ],
    )

    analysis = response.choices[0].message.content

    return (
        analysis.strip()
        if analysis and analysis.strip()
        else "No analysis could be generated."
    )


if __name__ == "__main__":
    example_data = lookup_sales_data(
        "Show me all the sales for store 1320 on November 1st, 2021"
    )
    print(example_data)

    print(
        analyze_sales_data(
            prompt="Which products have the highest sales?",
            data=example_data,
        )
    )

    tracer_provider.force_flush()



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
