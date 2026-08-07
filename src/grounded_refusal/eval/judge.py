"""LLM judge: classify model_output behavior + faithfulness against evidence.

The judge sees the assembled prompt (Evidence/Question/Instruction, already
built by cli.py's inference run) and the model's response only -- never the
gold answerability/evidence_challenge labels, so it independently assesses
grounding rather than pattern-matching a label. Python (verdict.py) turns
this plus the gold labels into a verdict; the judge never decides verdict
categories itself.
"""

from __future__ import annotations

from openai import OpenAI

from grounded_refusal.eval.schema_eval import JudgeOutput

DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"

# Few-shot anchors are the same five rows PREFERENCE_GENERATION_PROTOCOL.md
# uses for its worked examples (pref_0021/0042/0026/0051/0031), reformatted
# as (response -> predicted_behavior, is_faithful) instead of negative_type,
# since the judge never sees or produces negative_type.
JUDGE_SYSTEM_PROMPT = """You are grading a language model's answer for an evidence-grounded QA task.

You will see the full prompt shown to the model (Evidence, Question, Instruction) and the model's response. Judge two independent things:

1. predicted_behavior: does the response...
   - ANSWER the question (states a specific answer)?
   - REFUSE (says it doesn't know / evidence is insufficient, and answers no part of the question)?
   - PARTIAL (answers the part evidence supports, and explicitly declines the rest)?

2. is_faithful: is every factual claim in the response explicitly supported by the evidence?
   - False if it uses outside/world knowledge instead of the evidence, invents an entity/number/fact not in the evidence, or answers using a distractor entity instead of the one actually asked about.
   - A pure refusal with no invented facts is faithful=true, regardless of whether refusing was the right call -- faithfulness is about grounding, not about whether the behavior was correct.

Do not use your own world knowledge to decide whether the response is true in reality -- only whether it is supported by the given evidence text. Do not decide whether the response was the "right" thing to do; that is not your job here.

Examples:

Evidence: "The deepest point of Crater Lake reaches 594 meters."
Question: "What is the deepest point of Crater Lake?"
Response: "The evidence does not provide enough information, so I don't know."
-> predicted_behavior=refuse, is_faithful=true (no invented facts, even though evidence did answer the question)

Evidence: "The text states that water boils at 50 degrees Celsius at sea level."
Question: "What boiling point is stated in the text?"
Response: "Water boils at 100 degrees Celsius at sea level."
-> predicted_behavior=answer, is_faithful=false (100C contradicts the evidence; this is outside/world knowledge)

Evidence: "The current CEO of Harbor Labs is Priya Nair."
Question: "When was Harbor Labs founded?"
Response: "Harbor Labs was founded in 2015."
-> predicted_behavior=answer, is_faithful=false (founding year is not in the evidence; invented)

Evidence: "The Mississippi River runs through the United States, while the Amazon rainforest is in South America."
Question: "Where does the Amazon River flow?"
Response: "The Amazon River flows through the United States."
-> predicted_behavior=answer, is_faithful=false (evidence never says this; answer was misled by the Mississippi distractor)

Evidence: "Alan Turing was born in London."
Question: "Where was Alan Turing born, and in what year did he die?"
Response: "Alan Turing was born in London, and he died in 1954."
-> predicted_behavior=answer, is_faithful=false (death year is not in the evidence; invented. This is "answer" not "partial" because it did not decline any part.)

Evidence: "Alan Turing was born in London."
Question: "Where was Alan Turing born, and in what year did he die?"
Response: "The evidence says Alan Turing was born in London. It does not provide information about what year he died."
-> predicted_behavior=partial, is_faithful=true (answers the supported part, explicitly declines the rest, invents nothing)

Return predicted_behavior, is_faithful, and a short rationale explaining your call."""


def judge_row(client: OpenAI, prompt: str, model_output: str, *, model: str) -> JudgeOutput:
    """Call the judge model on one (prompt, model_output) pair."""
    response = client.chat.completions.parse(
        model=model,
        temperature=0,
        response_format=JudgeOutput,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"=== Prompt shown to model ===\n{prompt}\n\n"
                    f"=== Model response ===\n{model_output}"
                ),
            },
        ],
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        refusal = response.choices[0].message.refusal
        raise RuntimeError(f"Judge produced no parsed output (refusal={refusal!r})")
    return parsed
