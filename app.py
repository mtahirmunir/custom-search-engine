import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Load Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="DuckDuckGo")

# Title and description
st.title("🔎 LangChain - Chat with Search")
st.markdown("""
This app allows you to perform searches across different platforms (DuckDuckGo, Arxiv, Wikipedia) using LangChain. 
Select the search engine from the dropdown and interact with the chatbot.
""")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Dropdown for selecting the search engine
search_engine = st.sidebar.selectbox(
    "Select Search Engine",
    ["DuckDuckGo", "Arxiv", "Wikipedia"]
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Handle user input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Determine the selected tool
    if search_engine == "DuckDuckGo":
        selected_tool = search
    elif search_engine == "Arxiv":
        selected_tool = arxiv
    else:
        selected_tool = wiki

    # Initialize the selected tool and respond
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [selected_tool]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False  # Suppress detailed interaction logs
    )

    with st.chat_message("assistant"):
        try:
            # Execute the agent's response and display it
            response = search_agent.run(st.session_state.messages)
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
