# Agent_Evaluating

uv venv --python 3.12
source .venv/bin/activate

uv pip install \
  openai pandas duckdb pyarrow pydantic ipython \
  arize-phoenix arize-phoenix-otel \
  openinference-instrumentation-openai

phoenix serve