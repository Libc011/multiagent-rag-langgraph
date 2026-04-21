import asyncio
import os
from operator import add
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain.agents import create_agent  # 替换 create_react_agent

from config.load_key import load_key

nodes = ["travel", "joke", "song", "other"]

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=load_key("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=60,      # 默认太短就容易炸
    max_retries=3,   # 自动重试

)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str


async def supervisor_node(state: State):
    print(">>> supervisor_node")
    print({"node": ">>>> supervisor_node"})

    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
    如果用户的问题是和旅游路线规划相关的，那就返回travel。
    如果用户的问题是希望讲一个笑话，那就返回joke。
    如果用户的问题是希望写歌、写歌词、押韵歌词创作，那就返回song。
    如果是其他的问题，返回other。
    除了这几个选项外，不要返回任何其他的内容。"""

    user_text = state["messages"][-1].content if state.get("messages") else ""

    # 已有结果就结束
    if state.get("type") in nodes:
        print({"supervisor_step": f"已经获得{state['type']}智能体处理结果"})
        return {"type": END}

    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text},
    ]
    response = await llm.ainvoke(prompts)
    type_res = response.content.strip().lower()
    print({"supervisor_step": f"问题分类结果：{type_res}"})

    if type_res in nodes:
        return {"type": type_res}
    return {"type": "other"}


async def other_node(state: State):
    print(">>> other_node")
    print({"node": ">>>> other_node"})
    return {
        "messages": [AIMessage(content="我暂时无法回答这个问题")],
        "type": "other",
    }


async def travel_node(state: State):
    print(">>> travel_node")
    print({"node": ">>>> travel_node"})

    system_prompt = "你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游路线规划。请用中文回答，并返回一个不超过150字的行程表"
    user_text = state["messages"][-1].content if state.get("messages") else ""

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    client = MultiServerMCPClient(
        {
            "amap-maps": {
                "command": "npx",
                "args": ["-y", "@amap/amap-maps-mcp-server"],
                "env": {
                    "AMAP_MAPS_API_KEY": load_key("AMAP_MAPS_API_KEY"),
                },
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent(model=llm, tools=tools)

    response = await agent.ainvoke({"messages": [HumanMessage(content=user_text)]})
    final_text = response["messages"][-1].content
    print({"travel_result": final_text})

    return {"messages": [AIMessage(content=final_text)], "type": "travel"}


async def joke_node(state: State):
    print(">>> joke_node")
    print({"node": ">>>> joke_node"})
    user_text = state["messages"][-1].content if state.get("messages") else ""

    system_prompt = "你是一个笑话大师，根据用户的问题，写一个关于无畏契约的不超过100字的笑话"
    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    response = await llm.ainvoke(prompts)
    return {"messages": [AIMessage(content=response.content)], "type": "joke"}


async def song_node(state: State):
    print(">>> song_node")
    print({"node": ">>>> song_node"})
    user_text = state["messages"][-1].content if state.get("messages") else ""

    # 从你现有 Redis 对联库里取“押韵/句式参考”
    refs = []
    try:
        from rag_redis import search_couplets
        refs = search_couplets(user_text, k=5)
    except Exception as e:
        print({"song_rag_warn": f"检索失败，降级直出: {e}"})

    ref_text = "\n".join(
        [f"{i+1}. {u} | {d}" for i, (_, u, d) in enumerate(refs)]
    ) if refs else "（无参考语料）"

    system_prompt = """你是中文歌词创作助手。
请根据用户主题写歌词，要求：
1）输出结构：主歌A(4行) + 副歌(4行)
2）每行尽量 8-14 个字
3）副歌4行的尾字尽量同韵（押韵）
4）语言自然，避免文言腔过重
5）不要解释，只输出歌词正文"""

    user_prompt = f"""用户需求：{user_text}

可参考的押韵/句式语料（来自对联库，参考其韵律，不沿用文言表达）：
{ref_text}

请创作歌词："""

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {"messages": [AIMessage(content=response.content)], "type": "song"}


def routing_func(state: State):
    t = state.get("type")
    if t == "travel":
        return "travel_node"
    elif t == "joke":
        return "joke_node"
    elif t == "song":
        return "song_node"
    elif t == END:
        return END
    else:
        return "other_node"


builder = StateGraph(State)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("song_node", song_node)
builder.add_node("other_node", other_node)

builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges(
    "supervisor_node",
    routing_func,
    path_map=["travel_node", "joke_node", "song_node", "other_node", END],
)
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("song_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer =checkpointer)


async def main():
    config = {"configurable": {"thread_id": "1"}}
    final_state = None

    # 用 values，拿到每一步完整状态
    async for state in graph.astream(
        {"messages": [HumanMessage(content="帮我写一段emo情歌的歌词")]},
        config=config,
        stream_mode="values",
    ):
        final_state = state  # 始终保留最新状态

    # 从最终状态里提取最后一条 AI 消息
    if final_state and "messages" in final_state:
        for m in reversed(final_state["messages"]):
            if getattr(m, "type", "") == "ai":
                print("\n=== FINAL ANSWER ===")
                print(m.content)
                return

    print("未找到AI输出，请检查节点是否向 state['messages'] 追加了 AIMessage。")

if __name__ == "__main__":
    asyncio.run(main())
