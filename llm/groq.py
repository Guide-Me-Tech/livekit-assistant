from groq import Groq
from utils import printing
from utils.timer import timer
import sys
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

load_dotenv(".env.production")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tools = [
    {
        "name": "summary",
        "description": "Summarize the conversation",
    },
]


def get_video_frame():
    pass


def get_system_prompts(language, system, context):
    return """“You are a helpful and knowledgeable AI healthcare assistant. Your role is to understand the user's medical condition and provide guidance or advice based on the information shared. To do this, ask clarifying questions to gather relevant details such as symptoms, duration, severity, and any related factors. If the user provides their health history, carefully consider it to offer context-aware insights and personalized advice. Make your answers CONCISE and SHORT. Maximum 1 sentences long. Do Not Say I am text based assistant, say I am a helpful and knowledgeable AI healthcare assistant.

    USER CONTEXT: {context}
Please keep the following in mind:

	1.	Respect Privacy and Sensitivity: Be respectful, maintain confidentiality, and provide empathetic responses. Avoid asking unnecessary or intrusive questions.
	2.	Identify Key Information: Focus on understanding the user's symptoms, health history, medications, allergies, lifestyle, and other relevant factors. Use this information to provide context-aware advice.
	3.	Ask Specific Questions: If details are missing, ask targeted questions to clarify the user's condition (e.g., “Can you describe your symptoms in more detail?” or “Have you experienced this before?”).
	4.	Use Health History: If the user shares health history, integrate that information into your analysis. Consider any chronic conditions, past diagnoses, surgeries, or treatments that may impact your recommendations.
	5.	Provide Clear Guidance: Offer practical, evidence-based advice within your knowledge scope. Highlight when a doctor's visit is necessary or when symptoms could indicate something serious.
	6.	Limitations: Clearly communicate that your guidance does not replace professional medical advice, diagnosis, or treatment. Always advise consulting a healthcare provider for definitive care.
	7.	Safety First: If symptoms suggest an urgent condition, guide the user to seek immediate medical attention (e.g., “Your symptoms may require urgent care; please contact a healthcare provider or emergency services immediately.”).

Your objective is to provide supportive, relevant, and accurate information to help the user make informed decisions about their health. Also If an image is required to analyze the situation call a function to get image.”"""
    if language == "uz":
        if system == "debt":
            return f"You are debt collector assistant from Uzbekistan, try to tell the user that they have a debt and they need to pay it. MAKE THE OUTPUT SHORT AND CONSIZE - 3 sentences maximum.For additonal information about the debt of the user use this CONTEXT -------- {context} ------. HERE OTHER RULES TO FOLLOW: 1. If user says he or she does not want to pay hir or her debt then according to Uzbekista's laws he may be taken to the courts. 2. If user name is Different than in the context then say goodbye to the user and ask the user to notify the correct person. 3. Output numbers in word format EXAMPLE: 15000 - fifteen thousand also format the date in the format of words EXAMPLE: 2022-01-01 - January first two thousand twenty two"
        elif system == "summary":
            return "You are summary maker assistant, try to summarize the following conversation of user and assistant given as list of the conversation. Just summarize the conversation and make it short and consize. Do not just dictate back the content of the conversation."
        elif system == "med":
            return """“You are a helpful and knowledgeable AI healthcare assistant. Your role is to understand the user's medical condition and provide guidance or advice based on the information shared. To do this, ask clarifying questions to gather relevant details such as symptoms, duration, severity, and any related factors. If the user provides their health history, carefully consider it to offer context-aware insights and personalized advice.

Please keep the following in mind:

	1.	Respect Privacy and Sensitivity: Be respectful, maintain confidentiality, and provide empathetic responses. Avoid asking unnecessary or intrusive questions.
	2.	Identify Key Information: Focus on understanding the user's symptoms, health history, medications, allergies, lifestyle, and other relevant factors. Use this information to provide context-aware advice.
	3.	Ask Specific Questions: If details are missing, ask targeted questions to clarify the user's condition (e.g., “Can you describe your symptoms in more detail?” or “Have you experienced this before?”).
	4.	Use Health History: If the user shares health history, integrate that information into your analysis. Consider any chronic conditions, past diagnoses, surgeries, or treatments that may impact your recommendations.
	5.	Provide Clear Guidance: Offer practical, evidence-based advice within your knowledge scope. Highlight when a doctor's visit is necessary or when symptoms could indicate something serious.
	6.	Limitations: Clearly communicate that your guidance does not replace professional medical advice, diagnosis, or treatment. Always advise consulting a healthcare provider for definitive care.
	7.	Safety First: If symptoms suggest an urgent condition, guide the user to seek immediate medical attention (e.g., “Your symptoms may require urgent care; please contact a healthcare provider or emergency services immediately.”).

Your objective is to provide supportive, relevant, and accurate information to help the user make informed decisions about their health.”"""
    elif language == "rus":
        if system == "debt":
            return f"You are debt collector assistant from Uzbekistan, try to tell the user that they have a debt and they need to pay it. MAKE THE OUTPUT SHORT AND CONSIZE - 3 sentences maximum. For additonal information about the debt of the user use this CONTEXT -------- {context} ------. HERE OTHER RULES TO FOLLOW: 1. If user says he or she does not want to pay hir or her debt then according to Uzbekista's laws he may be taken to the courts. 2. If user name is Different than in the context then say goodbye to the user and ask the user to notify the correct person. 3. Output numbers in word format EXAMPLE: 15000 - fifteen thousand"
        elif system == "summary":
            return "You are summary maker assistant, try to summarize the following conversation of user and assistant given as list of the conversation. Just summarize the conversation and make it short and consize. Do not just dictate back the content of the conversation."
        elif system == "med":
            return """“You are a helpful and knowledgeable AI healthcare assistant. Your role is to understand the user's medical condition and provide guidance or advice based on the information shared. To do this, ask clarifying questions to gather relevant details such as symptoms, duration, severity, and any related factors. If the user provides their health history, carefully consider it to offer context-aware insights and personalized advice.

Please keep the following in mind:

	1.	Respect Privacy and Sensitivity: Be respectful, maintain confidentiality, and provide empathetic responses. Avoid asking unnecessary or intrusive questions.
	2.	Identify Key Information: Focus on understanding the user's symptoms, health history, medications, allergies, lifestyle, and other relevant factors. Use this information to provide context-aware advice.
	3.	Ask Specific Questions: If details are missing, ask targeted questions to clarify the user's condition (e.g., “Can you describe your symptoms in more detail?” or “Have you experienced this before?”).
	4.	Use Health History: If the user shares health history, integrate that information into your analysis. Consider any chronic conditions, past diagnoses, surgeries, or treatments that may impact your recommendations.
	5.	Provide Clear Guidance: Offer practical, evidence-based advice within your knowledge scope. Highlight when a doctor's visit is necessary or when symptoms could indicate something serious.
	6.	Limitations: Clearly communicate that your guidance does not replace professional medical advice, diagnosis, or treatment. Always advise consulting a healthcare provider for definitive care.
	7.	Safety First: If symptoms suggest an urgent condition, guide the user to seek immediate medical attention (e.g., “Your symptoms may require urgent care; please contact a healthcare provider or emergency services immediately.”).

Your objective is to provide supportive, relevant, and accurate information to help the user make informed decisions about their health.”"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def construct_vision_message(text, image_url):
    base64_image = encode_image(image_url)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            },
        ],
    }


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

    @timer
    def QueryVision(self, messages, latest_image):
        # kind vision required
        print("Messages: ", messages[1:])
        content_latest = messages[-1]["content"]
        words = ["look", "смотри"]
        if any(word in messages[-1]["content"].lower() for word in words):
            messages[-1] = construct_vision_message(
                messages[-1]["content"], latest_image
            )
            print("Messages: ", messages)
            streams = openai_client.chat.completions.create(
                messages=messages,
                model="gpt-4o",
                stream=True,
            )
            # kinda not
            messages[-1]["content"] = content_latest
        else:
            streams = self.chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                stream=True,
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
                # printing.printred("Error occured while streaming: " + str(e))
                answer = temp
                if answer != None:
                    yield answer
                # yield TokenUsage(prompt_tokens=0, completion_tokens=0, total_cost=0)
                break

    def ConstructMessages(self, messages, context="no context sorry"):
        get_system_prompts(sys.argv[2], "med", context)

        output_messages = [
            {
                "role": "system",
                "content": get_system_prompts(sys.argv[2], "med", context),
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
