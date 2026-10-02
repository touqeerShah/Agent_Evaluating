import warnings

from helper import get_phoenix_endpoint

warnings.filterwarnings("ignore")
from agents.agent import start_main_span, tools
import phoenix as px
import os
import json
from tqdm import tqdm
from phoenix.evals import TOOL_CALLING_PROMPT_TEMPLATE, llm_classify, OpenAIModel
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from openinference.instrumentation import suppress_tracing

import nest_asyncio

nest_asyncio.apply()
PROJECT_NAME = "evaluating-agent"
CLARITY_LLM_JUDGE_PROMPT = """
In this task, you will be presented with a query and an answer. Your objective is to evaluate the clarity 
of the answer in addressing the query. A clear response is one that is precise, coherent, and directly 
addresses the query without introducing unnecessary complexity or ambiguity. An unclear response is one 
that is vague, disorganized, or difficult to understand, even if it may be factually correct.

Your response should be a single word: either "clear" or "unclear," and it should not include any other 
text or characters. "clear" indicates that the answer is well-structured, easy to understand, and 
appropriately addresses the query. "unclear" indicates that some part of the response could be better 
structured or worded.
Please carefully consider the query and answer before determining your response.

After analyzing the query and the answer, you must write a detailed explanation of your reasoning to 
justify why you chose either "clear" or "unclear." Avoid stating the final label at the beginning of your 
explanation. Your reasoning should include specific points about how the answer does or does not meet the 
criteria for clarity.

[BEGIN DATA]
Query: {query}
Answer: {response}
[END DATA]
Please analyze the data carefully and provide an explanation followed by your response.

EXPLANATION: Provide your reasoning step by step, evaluating the clarity of the answer based on the query.
LABEL: "clear" or "unclear"
"""

agent_questions = [
    "What was the most popular product SKU?",
    "What was the total revenue across all stores?",
    "Which store had the highest sales volume?",
    "Create a bar chart showing total sales by store",
    "What percentage of items were sold on promotion?",
    "What was the average transaction value?",
]

for question in tqdm(agent_questions, desc="Processing questions"):
    try:
        ret = start_main_span([{"role": "user", "content": question}])
    except Exception as e:
        print(f"Error processing question: {question}")
        print(e)
        continue
print(get_phoenix_endpoint())
print(TOOL_CALLING_PROMPT_TEMPLATE)
## Router Evals using LLM-as-a-Judge

query = (
    SpanQuery()
    .where(
        # Filter for the `LLM` span kind.
        # The filter condition is a string of valid Python boolean expression.
        "span_kind == 'LLM'",
    )
    .select(question="input.value", tool_call="llm.tools")
)

# The Phoenix Client can take this query and return the dataframe.
tool_calls_df = px.Client().query_spans(query, project_name=PROJECT_NAME, timeout=None)
tool_calls_df = tool_calls_df.dropna(subset=["tool_call"])

tool_calls_df.head()

with suppress_tracing():
    tool_call_eval = llm_classify(
        dataframe=tool_calls_df,
        template=TOOL_CALLING_PROMPT_TEMPLATE.template[0].template.replace(
            "{tool_definitions}", json.dumps(tools).replace("{", '"').replace("}", '"')
        ),
        rails=["correct", "incorrect"],
        model=OpenAIModel(model="gpt-4o"),
        provide_explanation=True,
    )

tool_call_eval["score"] = tool_call_eval.apply(
    lambda x: 1 if x["label"] == "correct" else 0, axis=1
)

tool_call_eval.head()
px.Client().log_evaluations(
    SpanEvaluations(eval_name="Tool Calling Eval", dataframe=tool_call_eval),
)

query = (
    SpanQuery()
    .where("name =='generate_visualization'")
    .select(generated_code="output.value")
)
# Evaluating Python Code Gen (Tool 3 - Data Visualization Evals)
# The Phoenix Client can take this query and return the dataframe.
code_gen_df = px.Client().query_spans(query, project_name=PROJECT_NAME, timeout=None)

code_gen_df.head()


def code_is_runnable(output: str) -> bool:
    """Check if the code is runnable"""
    output = output.strip()
    output = output.replace("```python", "").replace("```", "")
    try:
        exec(output)
        return True
    except Exception as e:
        return False


code_gen_df["label"] = (
    code_gen_df["generated_code"]
    .apply(code_is_runnable)
    .map({True: "runnable", False: "not_runnable"})
)
code_gen_df["score"] = code_gen_df["label"].map({"runnable": 1, "not_runnable": 0})

code_gen_df.head()
px.Client().log_evaluations(
    SpanEvaluations(eval_name="Runnable Code Eval", dataframe=code_gen_df),
)
## Evaluating Analysis Clarity (Tool 2 - Data Analysis Evals)
query = (
    SpanQuery()
    .where("span_kind=='AGENT'")
    .select(response="output.value", query="input.value")
)

# The Phoenix Client can take this query and return the dataframe.
clarity_df = px.Client().query_spans(query, project_name=PROJECT_NAME, timeout=None)

clarity_df.head()
with suppress_tracing():
    clarity_eval = llm_classify(
        dataframe=clarity_df,
        template=CLARITY_LLM_JUDGE_PROMPT,
        rails=["clear", "unclear"],
        model=OpenAIModel(model="gpt-4o"),
        provide_explanation=True,
    )

clarity_eval["score"] = clarity_eval.apply(
    lambda x: 1 if x["label"] == "clear" else 0, axis=1
)

clarity_eval.head()
px.Client().log_evaluations(
    SpanEvaluations(eval_name="Response Clarity", dataframe=clarity_eval),
)

## Evaluating SQL Code Generation (Tool 1 - Database Lookup Evals)
SQL_EVAL_GEN_PROMPT = """
SQL Evaluation Prompt:
-----------------------
You are tasked with determining if the SQL generated appropiately answers a given instruction
taking into account its generated query and response.

Data:
-----
- [Instruction]: {question}
  This section contains the specific task or problem that the sql query is intended to solve.

- [Reference Query]: {query_gen}
  This is the sql query submitted for evaluation. Analyze it in the context of the provided
  instruction.

Evaluation:
-----------
Your response should be a single word: either "correct" or "incorrect".
You must assume that the db exists and that columns are appropiately named.
You must take into account the response as additional information to determine the correctness.

- "correct" indicates that the sql query correctly solves the instruction.
- "incorrect" indicates that the sql query correctly does not solve the instruction correctly.

Note: Your response should contain only the word "correct" or "incorrect" with no additional text
or characters.
"""
query = (
    SpanQuery()
    .where("span_kind=='LLM'")
    .select(
        query_gen="llm.output_messages",
        question="input.value",
    )
)

# The Phoenix Client can take this query and return the dataframe.
sql_df = px.Client().query_spans(query, project_name=PROJECT_NAME, timeout=None)
sql_df = sql_df[
    sql_df["question"].str.contains(
        "Generate an SQL query based on a prompt.", na=False
    )
]

sql_df.head()
with suppress_tracing():
    sql_gen_eval = llm_classify(
        dataframe=sql_df,
        template=SQL_EVAL_GEN_PROMPT,
        rails=["correct", "incorrect"],
        model=OpenAIModel(model="gpt-4o"),
        provide_explanation=True,
    )

sql_gen_eval["score"] = sql_gen_eval.apply(
    lambda x: 1 if x["label"] == "correct" else 0, axis=1
)

sql_gen_eval.head()
px.Client().log_evaluations(
    SpanEvaluations(eval_name="SQL Gen Eval", dataframe=sql_gen_eval),
)
