from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI
from utils.config_handler import rag_conf
# 从 langchain_openai 导入 LangChain 专用的 ChatOpenAI
from openai import OpenAI
from langchain_openai import ChatOpenAI

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return OllamaEmbeddings(
            model=rag_conf["embedding_model_name"],
            base_url="http://localhost:11434",
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
