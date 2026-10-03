import os
from openai import OpenAI
from pydantic import BaseModel, Field
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from skills.lookuo_sales_data import lookup_sales_data

MODEL = "qwen2.5:7b-instruct"
PROJECT_NAME = "tracing-agent"

# Use a plain URL, without Markdown link formatting.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=180.0,
)

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
# prompt template for step 1 of tool 3
CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""
# prompt template for step 2 of tool 3
CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""


# class defining the response format of step 1 of tool 3
class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")


# code for step 1 of tool 3
@tracer.chain()
def extract_chart_config(
    data: str,
    visualization_goal: str,
) -> dict:
    """Generate a validated chart configuration."""

    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
        data=data,
        visualization_goal=visualization_goal,
    )

    response = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a chart configuration as JSON. "
                    "Use only column names present in the supplied data. "
                    "Follow the requested chart type and axes."
                ),
            },
            {"role": "user", "content": formatted_prompt},
        ],
        response_format=VisualizationConfig,
    )

    message = response.choices[0].message

    if message.refusal:
        raise ValueError(f"Chart configuration refused: {message.refusal}")

    config = message.parsed

    if config is None:
        raise ValueError("The model did not return a chart configuration.")

    return {
        **config.model_dump(),
        "data": data,
    }

# code for step 2 of tool 3
@tracer.chain()
def create_chart(config: dict) -> str:
    """Create a chart based on the configuration"""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "")
    code = code.strip()

    return code


# code for tool 3
@tracer.tool()
def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate a visualization based on the data and goal"""
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    return code


example_data = lookup_sales_data(
    "Show me all the sales for store 1320 on November 1st, 2021"
)
print(example_data)
code = generate_visualization(
    example_data,
    "A bar chart of sales by product SKU. Put the product SKU on the x-axis and the sales on the y-axis.",
)
print(code)
