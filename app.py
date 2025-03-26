# Import necessary libraries
import os
import streamlit as st
import requests
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Set page config
st.set_page_config(
    page_title="Klacrt Crypto Assistant",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e1f5fe;
        border: 1px solid #b3e5fc;
    }
    .chat-message.assistant {
        background-color: #f1f1f1;
        border: 1px solid #e0e0e0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .suggestion-button {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        border: 1px solid #e0e0e0;
        background-color: #e8f0fe;
        cursor: pointer;
        text-align: center;
        transition: background-color 0.2s;
    }
    .suggestion-button:hover {
        background-color: #d2e3fc;
    }
    .crypto-info {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f9f9f9;
        margin-top: 1rem;
        border: 1px solid #e0e0e0;
    }
    .stMarkdown p {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# Set up environment variables
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key

# Initialize the Groq LLM and search tools
@st.cache_resource
def get_llm_with_search():
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.1,
    )
    
    # Initialize DuckDuckGo search
    search_tool = DuckDuckGoSearchRun()
    
    # Create a system prompt for using search
    system_prompt = """You are Klacrt, a helpful crypto assistant. When you don't know the answer or need current information, 
    use the search tool to find relevant information. Respond in a clear, concise manner.
    
    When searching:
    1. Formulate specific search queries about crypto
    2. Analyze the search results carefully
    3. Provide attribution for information you found through search
    4. If search doesn't yield relevant results, acknowledge limitations
    
    For crypto-specific questions, prioritize searching for recent and accurate information."""
    
    # Create conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # Initialize agent
    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        max_iterations=3
    )
    
    return agent, llm

# Initialize regular LLM for simple queries
@st.cache_resource
def get_llm():
    return ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.1,
    )

# Define crypto data functions
def get_crypto_price(symbol: str) -> Dict:
    """Get the current price and 24h change for a cryptocurrency by symbol (e.g., BTC, ETH)."""
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd&include_24hr_change=true")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_crypto_market_data(limit: int = 10) -> List[Dict]:
    """Get market data for top cryptocurrencies."""
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={limit}&page=1")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_crypto_historical_data(symbol: str, days: int = 30) -> Dict:
    """Get historical price data for a cryptocurrency."""
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart?vs_currency=usd&days={days}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def search_crypto_info(query: str):
    """Search for crypto information using DuckDuckGo."""
    try:
        agent, _ = get_llm_with_search()
        result = agent.run(f"crypto {query}")
        return result
    except Exception as e:
        return f"Error searching for information: {str(e)}"

def process_message(message: str):
    """Process the user message and return appropriate response with data."""
    message = message.lower()
    
    # Handle price requests
    if "price" in message and any(crypto in message for crypto in ["bitcoin", "btc"]):
        data = get_crypto_price("bitcoin")
        price = data.get("bitcoin", {}).get("usd", "N/A")
        change = data.get("bitcoin", {}).get("usd_24h_change", "N/A")
        if isinstance(change, float):
            change = round(change, 2)
        return f"Bitcoin is currently trading at ${price} USD (24h change: {change}%)", data
    
    elif "price" in message and any(crypto in message for crypto in ["ethereum", "eth"]):
        data = get_crypto_price("ethereum")
        price = data.get("ethereum", {}).get("usd", "N/A")
        change = data.get("ethereum", {}).get("usd_24h_change", "N/A")
        if isinstance(change, float):
            change = round(change, 2)
        return f"Ethereum is currently trading at ${price} USD (24h change: {change}%)", data
    
    # Handle market data requests
    elif any(word in message for word in ["market", "top", "coins", "cryptocurrencies"]):
        data = get_crypto_market_data(5)
        response = "Top 5 cryptocurrencies by market cap:\n"
        for coin in data:
            price_change = coin.get('price_change_percentage_24h')
            if isinstance(price_change, float):
                price_change = round(price_change, 2)
            response += f"- {coin['name']} (${coin['current_price']} USD, {price_change}% 24h change)\n"
        return response, data
    
    # Check if it's a specific crypto question that might need search
    elif any(crypto_term in message for crypto_term in ["crypto", "blockchain", "token", "coin", "defi", "nft", "wallet", "exchange", "mining"]):
        try:
            # Use web search for specific crypto questions
            search_result = search_crypto_info(message)
            return search_result, None
        except Exception as e:
            # Fall back to regular LLM if search fails
            llm = get_llm()
            ai_response = llm.invoke(message)
            return ai_response.content, None
    
    # Default response using Groq LLM
    else:
        try:
            llm = get_llm()
            ai_response = llm.invoke(message)
            return ai_response.content, None
        except Exception as e:
            return f"I'm not sure about that. Can you ask about crypto prices or markets?", None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Klacrt crypto assistant. Ask me about cryptocurrency prices, market data, or trends."}
    ]

# Display chat header
st.title("💰 Klacrt Crypto Assistant")
st.markdown("Ask questions about cryptocurrency prices, market data, and get AI-powered insights.")

# Display message history
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "assistant"
    avatar = "👤" if message["role"] == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {role_class}">
        <div class="avatar">{avatar}</div>
        <div class="message">{message["content"]}</div>
    </div>
    """, unsafe_allow_html=True)

# Suggestion buttons
st.markdown("<div style='display: flex; flex-wrap: wrap; margin-bottom: 1rem;'>", unsafe_allow_html=True)
suggestion_buttons = [
    "What's the Bitcoin price?",
    "Show Ethereum price",
    "Top cryptocurrencies"
]

cols = st.columns(len(suggestion_buttons))
for i, suggestion in enumerate(suggestion_buttons):
    with cols[i]:
        if st.button(suggestion, key=f"suggestion_{i}"):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            response, data = process_message(suggestion)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Chat input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", key="input", placeholder="Ask about crypto...")
    submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process the message
        with st.spinner("Thinking..."):
            response, data = process_message(user_input)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display additional data if available
        if data:
            if "bitcoin" in data or "ethereum" in data:
                # Display crypto price data
                crypto_name = "Bitcoin" if "bitcoin" in data else "Ethereum"
                symbol = "bitcoin" if "bitcoin" in data else "ethereum"
                price = data.get(symbol, {}).get("usd", "N/A")
                change = data.get(symbol, {}).get("usd_24h_change", "N/A")
                if isinstance(change, float):
                    change = round(change, 2)
                change_color = "green" if change > 0 else "red"
                
                st.markdown(f"""
                <div class="crypto-info">
                    <h3>{crypto_name} Data</h3>
                    <p>Current Price: <strong>${price} USD</strong></p>
                    <p>24h Change: <span style="color:{change_color}"><strong>{change}%</strong></span></p>
                </div>
                """, unsafe_allow_html=True)
            
            elif isinstance(data, list) and len(data) > 0:
                # Display market data
                st.markdown("<div class='crypto-info'><h3>Top Cryptocurrencies</h3></div>", unsafe_allow_html=True)
                
                # Create a table with market data
                market_data = []
                for coin in data[:5]:
                    price_change = coin.get('price_change_percentage_24h')
                    if isinstance(price_change, float):
                        price_change = round(price_change, 2)
                    market_data.append({
                        "Rank": coin.get('market_cap_rank', 'N/A'),
                        "Name": coin.get('name', 'N/A'),
                        "Symbol": coin.get('symbol', 'N/A').upper(),
                        "Price (USD)": f"${coin.get('current_price', 'N/A')}",
                        "24h Change (%)": price_change,
                        "Market Cap (USD)": f"${coin.get('market_cap', 'N/A'):,}"
                    })
                
                st.table(market_data)
        
        st.rerun()
