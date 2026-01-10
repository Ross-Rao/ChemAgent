"""
自定义 Embedding 类，用于替代 AzureOpenAIEmbedding
使用 OpenAI 兼容的 API 端点
"""
from typing import Any, List, Optional
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from openai import OpenAI


class CustomEmbedding(BaseEmbedding):
    """
    自定义 Embedding 类，兼容 AzureOpenAIEmbedding 的接口
    使用 OpenAI 兼容的 API 端点来获取文本嵌入向量
    
    参数:
    - api_key: API 密钥
    - deployment_name: 模型部署名称 (为了兼容原有接口，实际使用 model)
    - azure_endpoint: 不使用 (为了兼容原有接口保留)
    - api_version: 不使用 (为了兼容原有接口保留)
    - model: 使用的嵌入模型名称，默认 "text-embedding-3-small"
    - base_url: OpenAI API 的基础 URL
    - embed_batch_size: 批量处理的大小，默认 10
    """
    
    api_key: str = Field(description="API key")
    deployment_name: Optional[str] = Field(
        default="text-embedding-3-small",
        description="Deployment name (for compatibility, maps to model)"
    )
    azure_endpoint: Optional[str] = Field(
        default=None,
        description="Azure endpoint (not used, for compatibility)"
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version (not used, for compatibility)"
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Model name for embeddings"
    )
    base_url: str = Field(
        default="http://35.220.164.252:3888/v1/",
        description="Base URL for OpenAI API"
    )
    embed_batch_size: int = Field(
        default=10,
        description="Batch size for embedding requests"
    )
    
    _client: OpenAI = PrivateAttr()
    
    def __init__(
        self,
        api_key: str,
        deployment_name: str = "text-embedding-3-small",
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: str = "http://35.220.164.252:3888/v1/",
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        初始化自定义 Embedding 类
        
        保持与 AzureOpenAIEmbedding 相同的参数接口
        """
        super().__init__(
            api_key=api_key,
            deployment_name=deployment_name,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model=model,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
            **kwargs
        )
        
        # 初始化 OpenAI 客户端
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    @classmethod
    def class_name(cls) -> str:
        """返回类名"""
        return "CustomEmbedding"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        获取单个查询文本的嵌入向量
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量列表
        """
        return self._get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量列表
        """
        try:
            response = self._client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"调用 OpenAI Embeddings API 时发生错误: {e}")
            raise
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取多个文本的嵌入向量（批量处理）
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表的列表
        """
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self.model
            )
            # 按照原始顺序返回嵌入向量
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"调用 OpenAI Embeddings API (批量) 时发生错误: {e}")
            raise
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        异步获取查询文本的嵌入向量
        注意: 当前实现为同步调用的包装
        """
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        异步获取文本的嵌入向量
        注意: 当前实现为同步调用的包装
        """
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        异步获取多个文本的嵌入向量
        注意: 当前实现为同步调用的包装
        """
        return self._get_text_embeddings(texts)


# 方便直接使用的函数
def get_embedding(text, model="text-embedding-3-small", 
                  api_key="您的中转key", 
                  base_url="http://35.220.164.252:3888/v1/"):
    """
    获取给定文本的嵌入向量的便捷函数
    
    参数:
    text (str or list[str]): 需要嵌入的文本。可以是单个字符串或字符串列表。
    model (str): 使用的嵌入模型。
    api_key (str): API 密钥
    base_url (str): API 基础 URL
    
    返回:
    list or list[list]: 嵌入向量列表
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        if isinstance(text, str):
            return response.data[0].embedding
        else:
            return [item.embedding for item in response.data]
    except Exception as e:
        print(f"调用 OpenAI Embeddings API 时发生错误: {e}")
        return None
