import os
import json

def main():
    os.makedirs("data", exist_ok=True)

    examples = [
        {
            "instruction": "What's going on with NVIDIA?",
            "output": (
                "NVIDIA is focusing on new AI GPUs for data centers, expanding production for next-gen chips, and investing in robotics and autonomous systems.\n\n"
                "Why it matters:\n"
                "- Faster AI training/inference for businesses\n"
                "- More competition in AI hardware\n"
                "- Enables breakthroughs in areas like medical imaging and drug research\n\n"
                "Tell me your use case (gaming, ML study, course, startup) and I’ll tailor it."
            ),
        },
        {
            "instruction": "What’s new in AI?",
            "output": (
                "AI progress right now is centered on better reasoning models, AI agents that can use tools, and cheaper deployment on smaller hardware.\n\n"
                "Why it matters:\n"
                "- More automation (coding, research, customer support)\n"
                "- Higher productivity for students and teams\n"
                "- More need for safety, evaluation, and governance\n\n"
                "If you say your level (beginner/intermediate), I can suggest what to learn next."
            ),
        },
        {
            "instruction": "Explain RAG and how it helps in education.",
            "output": (
                "RAG (Retrieval-Augmented Generation) means a chatbot searches trusted documents first, then writes an answer using that retrieved info.\n\n"
                "How it helps in education:\n"
                "- Q&A over lecture notes and PDFs\n"
                "- Reduces hallucinations by grounding answers in course materials\n"
                "- Supports citations and reading comprehension tasks"
            ),
        },
        {
            "instruction": "User: What is a large model?\nUser: can it be used for my course design?",
            "output": (
                "A large model (LLM) is a neural network trained on massive text data that can generate and reason over language.\n\n"
                "Yes — for course design you can use it to:\n"
                "- Generate lesson outlines, quizzes, and examples\n"
                "- Create a course Q&A bot (best with RAG)\n"
                "- Provide personalized feedback\n\n"
                "If you tell me your subject and student level, I’ll propose a course workflow."
            ),
        },
    ]

    path = "data/train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Dataset created: {path}  (examples: {len(examples)})")
    print("Tip: add many more examples (200–1000+) for best fine-tuning.")

if __name__ == "__main__":
    main()
