import pandas as pd
import duckdb
import os
from openai import OpenAI
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import Status, StatusCode
# Local Phoenix server
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

PROJECT_NAME = "tracing-agent"
MODEL = "qwen2.5:7b-instruct"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=300.0,
    max_retries=0,
)

tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint=(
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"].rstrip("/")
        + "/v1/traces"
    ),
    protocol="http/protobuf",
    batch=False,
    sampler=ALWAYS_ON,  # Record every trace during local development.
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)

# define the path to the transactional data
TRANSACTION_DATA_FILE_PATH = "data/Store_Sales_Price_Elasticity_Promotions_Data.parquet"
# prompt template for step 2 of tool 1
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""


def update_sql_gen_prompt(new_prompt):
    global SQL_GENERATION_PROMPT
    SQL_GENERATION_PROMPT = new_prompt


# code for step 2 of tool 1
def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=prompt, columns=columns, table_name=table_name
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    return response.choices[0].message.content


# code for tool 1
@tracer.tool()
def lookup_sales_data(prompt: str) -> str:
    """Implementation of sales data lookup from parquet file using SQL"""
    try:

        # define the table name
        table_name = "sales"

        # step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        # step 2: generate the SQL code
        sql_query = generate_sql_query(prompt, df.columns, table_name)
        # clean the response to make sure it only includes the SQL code
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")
        with tracer.start_as_current_span(
            "execute_sql_query",
            openinference_span_kind="chain",
        ) as span:
            span.set_input(sql_query)

            result = duckdb.sql(sql_query).df()

            span.set_output(value=result.to_string(index=False))
            span.set_status(StatusCode.OK)

        return result.to_string(index=False)
    except Exception as e:
        return f"Error accessing data: {str(e)}"


example_data = lookup_sales_data(
    "Show me all the sales for store 1320 on November 1st, 2021"
)
print(example_data)


def get_sql_gen_prompt():
    table_name = "sales"
    df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
    columns = df.columns

    return SQL_GENERATION_PROMPT.format(
        prompt="question", columns=columns, table_name=table_name
    )
