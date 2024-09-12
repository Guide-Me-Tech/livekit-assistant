from groq import Groq

import time
from utils import printing


class GroqLLMHandlers(Groq):

    # self __init__(self, mode)

    def QueryStreams(self, messages, model_selected="llama-3.1-8b-instant"):
        print("Messages: ", messages)
        streams = self.chat.completions.create(
            messages=messages, model=model_selected, stream=True
        )

        full_answer = ""
        answer = ""
        temp = ""
        i = 0
        for chunk in streams:
            try:
                c = chunk.choices[0].delta.content
                # print(f"\033[94m c: {c} \033[0m")
                full_answer += c
                temp += c
                if c == "." or c == "!" or c == "?" or c == ":" or c == "\n" or c == "":
                    answer = temp
                    # print("sending form here")
                    temp = ""
                    if len(answer) > 0:
                        # print()
                        yield answer
                i += 1
            except Exception as e:
                printing.printred("Error occured while streaming: " + str(e))
                answer = temp
                if answer != None:
                    yield answer
                # yield TokenUsage(prompt_tokens=0, completion_tokens=0, total_cost=0)
                break

    def ConstructMessages(self, messages, context="no context sorry"):
        # output_messages = [
        #     {
        #         "role": "system",
        #         "content": f"You are debt collector assistant, try to tell the user that they have a debt and they need to pay it. Make the output short and consize. For additonal information about the debt of the user use this context : {context}",
        #     }
        # ]
        output_messages = [
            {
                "role": "system",
                "content": f"Complete the user sentences",
            }
        ]
        for message in messages:
            output_messages.append(
                {
                    "content": message["content"],
                    "role": message["role"],
                }
            )
        return output_messages

    def GenerateSummary(self, messages):

        # new_messages = [
        #     {
        #         "role": "assistant",
        #         "content": "You are summary maker assistant, try to summarize the following conversation of user and assistant given as list of the conversation. Just summarize the conversation and make it short and consize. Do not just dictate back the content of the conversation.",
        #     }
        # ]
        new_messages = [
            {
                "role": "assistant",
                "content": "You are summary maker assistant, try to summarize the following conversation of user and assistant given as list of the conversation. Just summarize the conversation and make it short and consize. Do not just dictate back the content of the conversation.",
            }
        ]
        new_messages.append({"role": "user", "content": str(messages[1:])})
        print("New Messages: ", new_messages)
        completion = self.chat.completions.create(
            messages=new_messages,
            model="llama-3.1-8b-instant",
        )
        return completion.choices[0].message.content
