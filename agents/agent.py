import json
from opentelemetry.trace import StatusCode

from skills.lookuo_sales_data import (
    lookup_sales_data,
    client,
    tracer,
    tracer_provider,
    MODEL,
)
from skills.data_anaylysis import analyze_sales_data
from skills.data_visualizations import generate_visualization

SYSTEM_PROMPT = """
You answer questions about the Store Sales Price Elasticity
Promotions dataset.

Use lookup_sales_data to retrieve data before answering factual
questions about sales.

For analysis, call analyze_sales_data using the retrieved data.
For visualizations, call generate_visualization using the retrieved
data and the user's visualization goal.

Do not invent sales figures or column names.
If a tool fails, correct the request when possible or explain the failure.
"""  # Define tools/functions that can be called by the model


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


def run_agent(messages, max_iterations=100, return_details=False):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    else:
        messages = [dict(message) for message in messages]

    # Put system instructions first.
    system_messages = [
        message for message in messages if message.get("role") == "system"
    ]
    conversation = [message for message in messages if message.get("role") != "system"]

    messages = (
        system_messages or [{"role": "system", "content": SYSTEM_PROMPT}]
    ) + conversation

    total_tool_calls = 0

    for iteration in range(max_iterations):
        print(f"Agent iteration {iteration + 1}")

        with tracer.start_as_current_span(
            "router_call",
            openinference_span_kind="chain",
        ) as span:
            span.set_input(value=messages)

            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=messages,
                tools=tools,
            )

            assistant = response.choices[0].message
            tool_calls = assistant.tool_calls or []

            assistant_message = {
                "role": "assistant",
                "content": assistant.content or "",
            }

            if tool_calls:
                assistant_message["tool_calls"] = [
                    call.model_dump(exclude_none=True) for call in tool_calls
                ]

            messages.append(assistant_message)

            span.set_output(value=assistant_message)
            span.set_status(StatusCode.OK)

        if not tool_calls:
            answer = assistant.content or "No response could be generated."

            if return_details:
                return {
                    "response": answer,
                    "messages": messages,
                    "model_calls": iteration + 1,
                    "tool_calls": total_tool_calls,
                    "path_length": iteration + 1 + total_tool_calls,
                }

            return answer

        total_tool_calls += len(tool_calls)
        messages = handle_tool_calls(tool_calls, messages)

    raise RuntimeError(f"Agent reached the limit of {max_iterations} model calls.")


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


if __name__ == "__main__":
    result = start_main_span(
        [{"role": "user", "content": "Which stores did the best in 2021?"}]
    )
    print(result)
    tracer_provider.force_flush()
