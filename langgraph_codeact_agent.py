"""
Langgraph Codeact Agent from https://github.com/langchain-ai/langgraph-codeact
"""

import argparse
import builtins
import contextlib
import io
from typing import Any

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph_codeact import create_codeact

from tools.math_ops import (
    add,
    ceil,
    cos,
    divide,
    exponentiation,
    floor,
    log,
    multiply,
    radians,
    round,
    sin,
    sqrt,
    subtract,
)
from tools.sam2_model import predict_image_masks, predict_video_masks
from tools.ultralytics_tools import predict_bboxes_and_masks, predict_poses


def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # Store original keys before execution
    original_keys = set(_locals.keys())

    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, builtins.__dict__, _locals)
        # Get the output of the code from the stdout print statements
        result = f.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"
    except Exception as e:
        result = f"Error during execution: {repr(e)}"

    # Determine new variables created during execution
    new_keys = set(_locals.keys()) - original_keys
    new_vars = {key: _locals[key] for key in new_keys}
    return result, new_vars


def do_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_prompt", type=str, required=True, help="Text prompt for the agent"
    )
    parser.add_argument("--llm_model", type=str, required=True, help="LLM model to use")
    parser.add_argument(
        "--default_sam2_model",
        type=str,
        required=False,
        help="Default SAM2 model Hugging Face URL to use",
        default="facebook/sam2-hiera-large",
    )
    parser.add_argument(
        "--system_prompt", type=str, required=False, help="System prompt for the agent"
    )
    return parser.parse_args()


def main():
    args = do_parsing()
    print(args)

    model = init_chat_model(args.llm_model, model_provider="ollama")

    tools = [
        add,
        multiply,
        divide,
        subtract,
        sin,
        cos,
        radians,
        exponentiation,
        sqrt,
        ceil,
        floor,
        round,
        log,
        predict_image_masks,
        predict_video_masks,
        predict_bboxes_and_masks,
        predict_poses,
    ]

    code_act = create_codeact(model, tools, eval, prompt=args.system_prompt)
    agent = code_act.compile(checkpointer=MemorySaver())

    messages = [{"role": "user", "content": args.text_prompt}]

    # Stream the agent's response (token by token)
    for typ, chunk in agent.stream(
        {"messages": messages},
        stream_mode=["values", "messages"],
        config={"configurable": {"thread_id": 1}},
    ):
        if typ == "messages":
            print(chunk[0].content, end="")
        elif typ == "values":
            print("\n\n---answer---\n\n", chunk)


if __name__ == "__main__":
    main()
