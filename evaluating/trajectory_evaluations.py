from phoenix.client import Client
import warnings

warnings.filterwarnings("ignore")
from phoenix.client import Client
from phoenix.client.experiments import (
    run_experiment,
    evaluate_experiment,
    create_evaluator,
)
from phoenix.otel import register
import pandas as pd
from datetime import datetime
# import nest_asyncio

# nest_asyncio.apply()
from agents.agent import run_agent

px_client = Client(base_url="http://localhost:6006")
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
dataset = px_client.datasets.create_dataset(
    dataframe=convergence_df,
    name=f"convergence_questions-{now}",
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


def run_agent_and_track_path(input: dict) -> dict:
    result = run_agent(
        input["question"],
        return_details=True,
    )

    return {
        "response": result["response"],
        "path_length": result["path_length"],
        "model_calls": result["model_calls"],
        "tool_calls": result["tool_calls"],
        "messages": format_message_steps(result["messages"]),
    }


experiment = run_experiment(
    dataset=dataset,
    task=run_agent_and_track_path,
    experiment_name="Convergence Eval",
    experiment_description="Comparing agent execution lengths across paraphrases",
    client=px_client,
    timeout=600,
    retries=0,
)
outputs = [
    run["output"]
    for run in experiment["task_runs"]
    if not run.get("error") and isinstance(run.get("output"), dict)
]

experiment_df = pd.DataFrame(outputs)

valid_lengths = [
    output["path_length"] for output in outputs if output.get("path_length", 0) > 0
]

if not valid_lengths:
    raise RuntimeError("No completed runs with valid path lengths.")

baseline_path_length = min(valid_lengths)

print(f"Shortest observed path: {baseline_path_length}")


@create_evaluator(name="Convergence Eval", kind="CODE")
def evaluate_path_length(output: dict) -> float:
    if not isinstance(output, dict):
        return 0.0

    path_length = output.get("path_length", 0)

    if path_length <= 0:
        return 0.0

    return baseline_path_length / float(path_length)


experiment = evaluate_experiment(
    experiment=experiment,
    evaluators=[evaluate_path_length],
    client=px_client,
)
