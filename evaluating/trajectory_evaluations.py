import warnings

warnings.filterwarnings("ignore")
import phoenix as px
from phoenix.evals import OpenAIModel
from phoenix.experiments import run_experiment, evaluate_experiment
from phoenix.experiments.types import Example
from phoenix.experiments.evaluators import create_evaluator
from phoenix.otel import register
import pandas as pd
from datetime import datetime
import os
import nest_asyncio

nest_asyncio.apply()
from agents.agent import run_agent
from helper import get_openai_api_key, get_phoenix_endpoint

px_client = px.Client()
convergence_questions = [
    "What was the average quantity sold per transaction?",
    "What is the mean number of items per sale?",
    "Calculate the typical quantity per transaction",
    "What's the mean transaction size in terms of quantity?",
    "On average, how many items were purchased per transaction?",
    "What is the average basket size per sale?",
    "Calculate the mean number of products per purchase",
    "What's the typical number of units per order?",
    "What is the average number of products bought per purchase?",
    "Tell me the mean quantity of items in a typical transaction",
    "How many items does a customer buy on average per transaction?",
    "What's the usual number of units in each sale?",
    "What is the typical amount of products per transaction?",
    "Show the mean number of items customers purchase per visit",
    "What's the average quantity of units per shopping trip?",
    "How many products do customers typically buy in one transaction?",
    "What is the standard basket size in terms of quantity?",
]

convergence_df = pd.DataFrame({"question": convergence_questions})

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
dataset = px_client.upload_dataset(
    dataframe=convergence_df,
    dataset_name=f"convergence_questions-{now}",
    input_keys=["question"],
)


# helper method to format the output returned by the task
def format_message_steps(messages):
    """
    Convert a list of message objects into a readable format that shows the steps taken.

    Args:
        messages (list): A list of message objects containing role, content, tool calls, etc.

    Returns:
        str: A readable string showing the steps taken.
    """
    steps = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            steps.append(f"User: {message.get('content')}")
        elif role == "system":
            steps.append("System: Provided context")
        elif role == "assistant":
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    steps.append(f"Assistant: Called tool '{tool_name}'")
            else:
                steps.append(f"Assistant: {message.get('content')}")
        elif role == "tool":
            steps.append(f"Tool response: {message.get('content')}")

    return "\n".join(steps)


def run_agent_and_track_path(example: Example) -> str:
    messages = [{"role": "user", "content": example.input.get("question")}]
    ret = run_agent(messages)
    return {"path_length": len(ret), "messages": format_message_steps(ret)}


experiment = run_experiment(
    dataset,
    run_agent_and_track_path,
    experiment_name="Convergence Eval",
    experiment_description="Evaluating the convergence of the agent",
)
experiment.as_dataframe()

outputs = experiment.as_dataframe()["output"].to_dict().values()

# Will include the user and system messages
optimal_path_length = min(output.get('path_length') for output in outputs if output and output.get('path_length') is not None)
print(f"The optimal path length is {optimal_path_length}")
@create_evaluator(name="Convergence Eval", kind="CODE")
def evaluate_path_length(output: str) -> float:
    if output and output.get("path_length"):
        return optimal_path_length/float(output.get("path_length"))
    else:
        return 0

experiment = evaluate_experiment(experiment,
                            evaluators=[evaluate_path_length])