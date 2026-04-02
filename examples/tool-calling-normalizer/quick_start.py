from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm2_agent_loop import LFMAgentLoop


def get_time() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat() + "Z"


def add(x: int, y: int) -> int:
    return x + y


def run_example() -> None:
    model_name = "LiquidAI/LFM2.5-1.2B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    agent = LFMAgentLoop(
        model=model,
        tokenizer=tokenizer,
        tools=[get_time, add],
        system_prompt="You may call tools by returning a tool call.",
        force_json=True,
        max_turns=2,
        device="cpu",
    )

    for query in [
        "What is the current UTC time?",
        "Add 12 and 30 using the tool.",
    ]:
        print("QUERY:", query)
        print(agent.run(query))
        print("---")


if __name__ == "__main__":
    run_example()
