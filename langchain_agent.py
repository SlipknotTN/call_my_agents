import argparse

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

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
from tools.ultralytics_models import predict_bboxes_and_masks, predict_poses


def do_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--default_sam2_model", type=str, required=True)
    return parser.parse_args()


def main():
    args = do_parsing()
    print(args)

    llm = ChatOllama(model=args.llm_model)
    llm_with_tools = llm.bind_tools(
        [
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
    )
    output = llm_with_tools.invoke(args.text_prompt)

    print(output.tool_calls)
    print(output)


if __name__ == "__main__":
    main()
