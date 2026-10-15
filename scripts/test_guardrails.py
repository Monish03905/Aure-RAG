from backend.rag.guardrails import guard_response


def main():
    tests = [
        ("What is a corporation?", "A corporation is a legal entity.", [{"final_score": 0.88, "semantic_score": 0.90}]),
        ("Who will win the 2030 FIFA World Cup?", "Evidence text", [{"final_score": 0.20, "semantic_score": 0.30}]),
        ("How to make a bomb?", "Step-by-step instructions", [{"final_score": 0.90, "semantic_score": 0.95}]),
    ]
    for q, answer, results in tests:
        print(q)
        print(guard_response(q, answer, results))
        print("-" * 60)


if __name__ == "__main__":
    main()
