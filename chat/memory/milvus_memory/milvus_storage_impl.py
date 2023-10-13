from .milvus_memory import MilvusMemory
from ..base_storage import BaseStorage
import configs.config as config


class MilvusStorage(BaseStorage):
    '''Milvus向量存储记忆模块'''
    milvus_memory: MilvusMemory
    def __init__(self):

        self.milvus_memory = MilvusMemory()

    def search(self, query_text: str, user_name: str,character_name:str) -> list[str]:
        if  self.milvus_memory.is_exist(user_name,character_name):
            self.milvus_memory.load_milvus_collection(user_name,character_name)
            expr = f"user_name == '{user_name}'"
            # 查询记忆，并且使用 关联性 + 重要性 + 最近性 算法进行评分
            memories = self.milvus_memory.compute_relevance(query_text, config.VECTOR_SEARCH_TOP_K, expr=expr)
            self.milvus_memory.compute_recency(memories)
            self.milvus_memory.normalize_scores(memories)
            self.milvus_memory.release()

            # 排序获得最高分的记忆
            memories = sorted(memories, key=lambda m: m["total_score"], reverse=True)

            if len(memories) > 0:
                memories_text = [item['text'] for item in memories]
                memories_size = config.LONG_SEACH_MEMORY_LEN
                memories_text = memories_text[:memories_size] if len(memories_text) >= memories_size else memories_text
                return memories_text
            else:
                return []
        else:
            return []

    def pageQuery(self, page_num: int, page_size: int, owner: str) -> list[str]:
        offset = (page_num - 1) * page_size
        limit = page_size
        result = self.milvus_memory.pageQuery(
            offset=offset, limit=limit, expr=f"owner={owner}")
        self.milvus_memory.release()
        return result

    def save(self, pk: int, owner :str , character_name : str,  query_text: str,responce :str , importance_score: int) -> None:
        self.milvus_memory.insert_memory(pk=pk, text=query_text, user_name=owner, importance_score=importance_score,character_name=character_name)
        self.milvus_memory.release()

    def clear(self, owner: str) -> None:
        self.milvus_memory.loda()
        self.milvus_memory.clear(owner)
        self.milvus_memory.release()
