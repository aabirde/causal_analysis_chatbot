from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI
import os
import warnings
from langchain.cache import InMemoryCache
from dotenv import load_dotenv
load_dotenv()
llm1 = ChatTogether(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    together_api_key=os.getenv("TOGETHER_API_KEY"),
    model_kwargs={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 10000
    }
)

llm2 = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free",
    openai_api_key=os.getenv("OPEN_API_KEY"),
    max_tokens=10000,
    base_url="https://openrouter.ai/api/v1"
)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm4 = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.6,
    max_tokens=None,
)



llm1.cache = InMemoryCache()
llm4.cache = InMemoryCache()
warnings.filterwarnings('ignore')