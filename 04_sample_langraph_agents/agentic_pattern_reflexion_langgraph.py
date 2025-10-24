# https://medium.com/aimonks/reflection-agents-with-langgraph-agentic-llm-based-applications-87e43c27adc7

from langchain_core.messages import HumanMessage, SystemMessage
from decouple import config
from langchain_openai import ChatOpenAI

from typing import List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# generate agent
prompt = ChatPromptTemplate.from_messages(
    [
        #  this needs to be a tuple so do not forget the , at the end and do not include any , in-between
        (
            "system",
            "You are an AI assistant researcher tasked with researching on a variety of topics in a short summary of 5 paragraphs."
            " Generate the best research possible as per user request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(api_key=config("OPENAI_API_KEY"))


generate = prompt | llm


research = ""
request = HumanMessage(
    content="Research on climate change"
)

# test the single generate agent
for chunk in generate.stream({"messages": [request]}):
    print(chunk.content, end="")
    research += chunk.content


# reflect agent
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        #  this needs to be a tuple so do not forget the , at the end
        (
            "system",
            "You are a senior researcher"
            " Provide detailed recommendations, including requests for length, depth, style, etc."
            " to an asistant researcher to help improve this researches",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm

print("\n\n\n")
print("============Relect===============")
print("\n\n\n")

reflection = ""
for chunk in reflect.stream({"messages": [request, HumanMessage(content=research)]}):
    print(chunk.content, end="")
    reflection += chunk.content



# basic reflection agent in Python
import asyncio
from langgraph.graph import END, MessageGraph
from typing import List, Sequence
from langgraph.graph import MessageGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from decouple import config
from langchain_openai import ChatOpenAI

from typing import List


from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant researcher tasked with researching on a variety of topics in a short summary of 5 paragraphs."
            " Generate the best research possible as per user request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(api_key=config("OPENAI_API_KEY"))


generate = prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior researcher"
            " Provide detailed recommendations, including requests for length, depth, style, etc."
            " to an asistant researcher to help improve this researches",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect = reflection_prompt | llm


async def generation_node(state: Sequence[BaseMessage]):
    return await generate.ainvoke({"messages": state})


async def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}

    # First message is the original user request. We hold it the same for all nodes
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    res = await reflect.ainvoke({"messages": translated})

    # this will be treated as a feedback for the generator
    return HumanMessage(content=res.content)


builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        # End after 5 iterations
        return END
    return "reflect"


builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()


async def stream_responses():
    async for event in graph.astream(
        [
            HumanMessage(
                content="Research on climate change"
            )
        ],
    ):
        print(event)
        print("---")


asyncio.run(stream_responses())