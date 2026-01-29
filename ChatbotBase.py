"""
ChatbotBase.py
Base class template for the chatbot.

"""

class ChatbotBase:
    """Base class that all chatbot implementations should inherit from."""

    def __init__(self, name="TechBot"):
        self.name = name
        self.context = {}
        self.conversation_history = []

    def preprocess(self, user_input: str) -> str:
        return user_input.strip()

    def classify_intent(self, processed_input: str):
        return {"query": True}

    def retrieve_info(self, intent, processed_input: str):
        return {"content": ["I'm not sure how to respond."], "resources": []}

    def generate_response(self, retrieved_info: dict) -> str:
        content = retrieved_info.get("content", [])
        if isinstance(content, list):
            return "\n".join(content)
        return str(content)

    def respond(self, user_input: str) -> str:
        # Step 1: Preprocess
        processed = self.preprocess(user_input)

        # Step 2: Store in history
        self.conversation_history.append({"user": user_input, "processed": processed})

        # Step 3: Classify intent
        intent = self.classify_intent(processed)

        # Step 4: Retrieve info
        retrieved = self.retrieve_info(intent, processed)

        # Step 5: Generate response
        response = self.generate_response(retrieved)

        # Step 6: Store response
        self.conversation_history[-1]["response"] = response
        return response
