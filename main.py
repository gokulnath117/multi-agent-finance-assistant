# main.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from tools import search_tool, news_tool
from rag_store import add_to_rag, query_rag
from voice_io import get_voice_input, speak_text
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
tools = [search_tool, news_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are a highly intelligent and reliable Financial Assistant. You help users make informed financial decisions by analyzing historical stock data and scraping the latest financial news.
        You have access to two specialized tools:
        Search Tool Use this tool to retrieve and analyze historical stock price data for any public company or financial instrument.
        News Tool Use this tool to fetch the most recent and relevant news articles or headlines related to a company, industry, or financial topic.'''),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_query(user_input):
    result = agent_executor.invoke({"input": user_input})
    tool_output = result['output']
    print("\n📦 Tool Output:\n", tool_output)

    add_to_rag(tool_output)
    response = query_rag(user_input)

    print("\n🧠 Final Answer:\n", response)
    speak_text(response)

if __name__ == "__main__":
    query = get_voice_input()
    if query:
        run_query(query)