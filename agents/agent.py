from openai import OpenAI
import pandas as pd
import json
import duckdb
from pydantic import BaseModel, Field
from IPython.display import Markdown

from helper import get_openai_api_key

# initialize the OpenAI client
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
# Define tools/functions that can be called by the model
import json

from skills.data_Anaylysis import analyze_sales_data
from skills.data_visualizations import generate_visualization
from skills.lookuo_sales_data import lookup_sales_data

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Store Sales Price Elasticity Promotions dataset.
"""
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_sales_data",
            "description": "Look up data from Store Sales Price Elasticity Promotions dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The unchanged prompt that the user provided.",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sales_data",
            "description": "Analyze sales data to extract insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The lookup_sales_data tool's output.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The unchanged prompt that the user provided.",
                    },
                },
                "required": ["data", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate Python code to create data visualizations",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The lookup_sales_data tool's output.",
                    },
                    "visualization_goal": {
                        "type": "string",
                        "description": "The goal of the visualization.",
                    },
                },
                "required": ["data", "visualization_goal"],
            },
        },
    },
]

# Dictionary mapping function names to their implementations
tool_implementations = {
    "lookup_sales_data": lookup_sales_data,
    "analyze_sales_data": analyze_sales_data,
    "generate_visualization": generate_visualization,
}


# code for executing the tools returned in the model's response
@tracer.chain()
def handle_tool_calls(tool_calls, messages):

    for tool_call in tool_calls:
        function = tool_implementations[tool_call.function.name]
        function_args = json.loads(tool_call.function.arguments)
        result = function(**function_args)
        messages.append(
            {"role": "tool", "content": result, "tool_call_id": tool_call.id}
        )

    return messages


def run_agent(messages):
    print("Running agent with messages:", messages)
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not any(
        isinstance(message, dict) and message.get("role") == "system"
        for message in messages
    ):
        system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
        messages.append(system_prompt)

    while True:
        # Router Span
        print("Starting router call span")
        with tracer.start_as_current_span(
            "router_call",
            openinference_span_kind="chain",
        ) as span:
            span.set_input(value=messages)

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
            )
            messages.append(response.choices[0].message.model_dump())
            tool_calls = response.choices[0].message.tool_calls
            print("Received response with tool calls:", bool(tool_calls))
            span.set_status(StatusCode.OK)

            if tool_calls:
                print("Starting tool calls span")
                messages = handle_tool_calls(tool_calls, messages)
                span.set_output(value=tool_calls)
            else:
                print("No tool calls, returning final response")
                span.set_output(value=response.choices[0].message.content)
                return response.choices[0].message.content


def start_main_span(messages):
    print("Starting main span with messages:", messages)

    with tracer.start_as_current_span(
        "AgentRun", openinference_span_kind="agent"
    ) as span:
        span.set_input(value=messages)
        ret = run_agent(messages)
        print("Main span completed with return value:", ret)
        span.set_output(value=ret)
        span.set_status(StatusCode.OK)
        return ret


result = start_main_span(
    [{"role": "user", "content": "Which stores did the best in 2021?"}]
)
