# agent_logic.py
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import search_tool, news_tool
from rag_store import add_to_rag, query_rag
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

tools = [news_tool, search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant. Use tools to retrieve stock and news data, then answer."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_query(user_input: str) -> str:
    tool_outputs = agent_executor.invoke({"input": user_input})
    result_text = tool_outputs.get("output", "")
    if result_text:
        add_to_rag(result_text)

    return query_rag(user_input)