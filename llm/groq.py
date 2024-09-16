from groq import Groq

import time
from utils import printing
from utils.timer import timer
import sys


class GroqLLMHandlers(Groq):

    # self __init__(self, mode)
    @timer
    def QueryStreams(self, messages, model_selected="llama-3.1-70b-versatile"):
        print("Messages: ", messages[1:])
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
        if sys.argv[2] == "uz":
            output_messages = [
                {
                    "role": "system",
                    "content": f"You are debt collector assistant from Uzbekistan, try to tell the user that they have a debt and they need to pay it. MAKE THE OUTPUT SHORT AND CONSIZE - 3 sentences maximum.For additonal information about the debt of the user use this CONTEXT -------- {context} ------. HERE OTHER RULES TO FOLLOW: 1. If user says he or she does not want to pay hir or her debt then according to Uzbekista's laws he may be taken to the courts. 2. If user name is Different than in the context then say goodbye to the user and ask the user to notify the correct person. 3. Output numbers in word format EXAMPLE: 15000 - fifteen thousand also format the date in the format of words EXAMPLE: 2022-01-01 - January first two thousand twenty two",
                }
            ]
        else:
            print("Constructing messages for RUS")
            output_messages = [
                {
                    "role": "system",
                    "content": f"You are debt collector assistant from Uzbekistan, try to tell the user that they have a debt and they need to pay it. MAKE THE OUTPUT SHORT AND CONSIZE - 3 sentences maximum. For additonal information about the debt of the user use this CONTEXT -------- {context} ------. HERE OTHER RULES TO FOLLOW: 1. If user says he or she does not want to pay hir or her debt then according to Uzbekista's laws he may be taken to the courts. 2. If user name is Different than in the context then say goodbye to the user and ask the user to notify the correct person. 3. Output numbers in word format EXAMPLE: 15000 - fifteen thousand",
                }
            ]

        # output_messages = [
        #     {
        #         "role": "system",
        #         "content": f"You are ",
        #     }
        # ]
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
