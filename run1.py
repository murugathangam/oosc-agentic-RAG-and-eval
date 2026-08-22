import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests

def ask(question: str):
    response = requests.post("http://localhost:8000/agent_chat", json={"message": question})
    data = response.json()

    print("=" * 90)
    print(f"Q: {question}")
    print("=" * 90)

    print("\n--- ANSWER ---")
    print(data["answer"])

    print("\n--- TRAJECTORY ---")
    for step in data["trajectory"]:
        print(f"  Step {step['step']}: \"{step['query']}\"")
        print(f"    → retrieved: {step['retrieved_ids']}  (best_distance={step['best_distance']:.3f})")
        flags = []
        if step["redundant_step"]:
            flags.append("REDUNDANT")
        if step["loop_detected"]:
            flags.append("LOOP")
        if flags:
            print(f"    ⚠ flags: {', '.join(flags)}")
        print(f"    drift_from_original: {step['drift_from_original']:.3f}")

    m = data["metrics"]
    print("\n--- METRICS ---")
    print(f"  Total steps:            {m['total_steps']}")
    print(f"  Unnecessary steps:      {m['unnecessary_steps']}")
    print(f"  Loop detected:          {m['loop_detected']}")
    print(f"  Drift compounded:       {m['drift_compounded']:.3f}")
    print(f"  Avg retrieval distance: {m['avg_retrieval_distance']:.3f}")
    print(f"  Efficiency score:       {m['efficiency_score']}")

    print("\n--- FAITHFULNESS ---")
    print(f"  {data['faithfulness']['raw_verdict']}")

    print("\n")


questions = [
    "When did Faraday discover electromagnetic induction?",
    "How did Maxwell's equations lead to the realization that light is an electromagnetic wave?",
    "What is the Aharonov-Bohm effect?",
]

for q in questions:
    ask(q)