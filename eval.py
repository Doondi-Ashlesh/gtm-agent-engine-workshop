import os

from langsmith import evaluate

from gtm_agent.gtm_agent import run_agent


def evaluation_target(inputs: dict) -> dict:
    user_message = inputs["messages"][0]["content"]

    result = run_agent(
        user_message,
        user_id=inputs.get("user_id"),
        thread_id=inputs.get("thread_id"),
    )
    return {"output": result["reply"], "run_id": result["run_id"]}


def main() -> None:
    # Hard-fails if DATASET_NAME is not set.
    dataset = os.environ["DATASET_NAME"]

    evaluate(
        evaluation_target,
        data=dataset,
        experiment_prefix="baseline",
    )


if __name__ == "__main__":
    main()
