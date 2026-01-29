#!/usr/bin/env python3
"""
run.py
Main script to run TechBot Pro.
"""

from chatbot import TechBot


def main():
    print("\n" + "=" * 60)
    print("🤖 TECHBOT PRO - Advanced Tech Information Assistant")
    print("=" * 60)

    try:
        bot = TechBot(name="TechBot Pro")
    except Exception as e:
        print(f"❌ Failed to initialize bot: {e}")
        return

    print("\n💡 Tips:")
    print("  • Ask any tech question: 'What is ChatGPT?', 'Explain RAG'")
    print("  • Get news: 'What's new in AI?', 'Latest OpenAI news'")
    print("  • Add 'with links' if you want sources")
    print("  • Commands: help, profile, progress, done <keyword>, quit")
    print("-" * 60)

    user_id = input("\nEnter your user ID (or press Enter for default): ").strip()
    if not user_id:
        user_id = "default_user"

    bot.context["current_user"] = user_id

    profile = bot.get_user_profile(user_id)

    # Offer profile setup for new user
    if profile["knowledge_level"] == "beginner" and not profile["interests"]:
        print(f"\n👋 Welcome, {user_id}!")
        setup = input("Would you like to set up your profile? (y/n): ").lower().strip()

        if setup == "y":
            level = input("Knowledge level (beginner/intermediate/advanced): ").lower().strip()
            interests = input("Interests (comma-separated, e.g., ai,web,cloud): ").lower().strip()
            goals = input("Learning goals: ").strip()

            bot.set_user_profile(
                user_id=user_id,
                level=level if level in ["beginner", "intermediate", "advanced"] else "beginner",
                interests=[i.strip() for i in interests.split(",") if i.strip()],
                goals=[goals] if goals else [],
            )
            print("✅ Profile created!")

    print(f"\nHello {user_id}! Ask me about **any tech topic**.\n")
    print("Examples: 'What is ChatGPT?', 'Explain AI agents', 'What's new with OpenAI?'\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower().strip()

            if cmd == "quit":
                print("\n👋 Goodbye! Keep learning!")
                break

            if cmd == "help":
                print("\n" + "=" * 50)
                print("📋 **Commands:**")
                print("  help        - show this help")
                print("  profile     - show your profile")
                print("  progress    - show your learning progress")
                print("  done <kw>   - mark suggested resources done")
                print("  quit        - exit")
                print("")
                print("💬 **Question Examples:**")
                print("  what is ChatGPT?")
                print("  explain RAG")
                print("  tell me about OpenAI")
                print("  what's new in AI?")
                print("  latest NVIDIA news")
                print("  how does GPT work?")
                print("  what are AI agents?")
                print("")
                print("📚 **Get Resources:**")
                print("  learn about AI")
                print("  python resources for beginners")
                print("=" * 50)
                continue

            if cmd == "profile":
                p = bot.get_user_profile(user_id)
                print("\n👤 **Your Profile:**")
                print(f"  Level: {p['knowledge_level']}")
                print(f"  Interests: {', '.join(p['interests']) if p['interests'] else 'Not set'}")
                print(f"  Goals: {', '.join(p['goals']) if p['goals'] else 'Not set'}")
                print("")
                continue

            if cmd == "progress":
                print("\n" + bot.get_progress_report(user_id))
                print("")
                continue

            if cmd.startswith("done "):
                keyword = user_input[5:].strip()
                print("\n" + bot.mark_resource_done(user_id, keyword))
                print("")
                continue

            # Normal chat
            response = bot.respond(user_input)
            print(f"\n{bot.name}: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            print("Please try rephrasing your question.\n")


if __name__ == "__main__":
    main()
