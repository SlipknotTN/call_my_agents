"""
Agent implementation without frameworks

In order to run this script you need to host the LLM locally with ollama

Steps:
- Install ollama into the host machine https://ollama.com/download/linux
- Pull the model you want to use with "ollama pull <model_name>", e.g. "ollama pull llama3.2"
- [Optional] Install open-webui into the host machine to try the models with a ChatGPT browser like interface
  https://github.com/open-webui/open-webui?tab=readme-ov-file#installation-via-python-pip-
"""

import argparse
import json
import logging
import random

from ollama import ChatResponse, chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def do_parsing():
    parser = argparse.ArgumentParser(
        description="Vanilla agent implementation without frameworks"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="mistral",
        choices=["mistral", "llama3.2", "qwen2.5-coder:3b"],
        help="Ollama model name. Available list: https://ollama.com/library",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        required=False,
        default=None,
        help="System prompt",
    )
    parser.add_argument("--question", type=str, required=True, help="Problem statement")
    parser.add_argument(
        "--tools_source_files",
        type=str,
        nargs="+",
        required=True,
        help="Source files to use as tools",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    logger.info(args)

    question = args.question
    system_prompt = args.system_prompt

    tools_code = ""
    for tool_source_file in args.tools_source_files:
        with open(tool_source_file, "r") as f:
            tool_source_code = f.read()
        tools_code += f"## File {tool_source_file}\n\n"  # This helps the LLM to understand the necessary imports, the paths need to be relative to the current file
        tools_code += "```python\n"
        tools_code += tool_source_code
        tools_code += "```\n\n"

    messages = []
    if system_prompt is not None and system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    # It is more effective to pass the tools code as user prompt instead of system prompt (at least for qwen2.5-coder:3b)
    messages.append({"role": "user", "content": tools_code + "\n\n" + question})

    response: ChatResponse = chat(
        model=args.llm_model,
        messages=messages
    )
    logger.info(f"Answer: {response.message.content}")


if __name__ == "__main__":
    main()
