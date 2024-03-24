import openai

SYSTEM_PROMPT = """You are an exciting sports commentator for a ping pong match, with a high capacity for roasting the participants. Please keep commentary short: i.e., to about one sentence, as if your responses will be spoken between points."""
I = openai.OpenAI()

class Commentary:
    def __init__(self):
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def reset(self):
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_commentary(self, who_scored, num_hits):
        hits_str = 'hits' if num_hits != 1 else 'hit'
        self.history.append({"role": "user", "content": f"Player {who_scored} scored. This rally had {num_hits} {hits_str}."})
        cmpl = I.chat.completions.create(messages=self.history, model='gpt-4-turbo-preview') # type: ignore
        response_text = cmpl.choices[0].message.content

        assert response_text is not None, "Received no response text."

        return response_text
