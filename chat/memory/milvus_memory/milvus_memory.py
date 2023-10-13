# 导入所需模块
import os

from pymilvus import DataType, FieldSchema, CollectionSchema, Collection, connections
from sentence_transformers import SentenceTransformer
import time
import configs.config as config


# _COLLECTION_NAME = "virtual_wife_memory_v1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MilvusMemory():
    collection: Collection
    schema: CollectionSchema
    def __init__(self):
        connections.connect(host=config.milvus_host,port=config.milvus_port,user=config.milvus_user,password=config.milvus_password,db_name=config.milvus_db_name,)
        # 定义记忆Stream集合schema、创建记忆Stream集合
        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="owner", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="importance_score", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,dim=768),  # 文本embedding向量
        ]
        # 创建索引
        self.index = {"index_type": "IVF_SQ8","metric_type": "L2","params": {"nlist": 768},}
        self.schema = CollectionSchema(self.fields, config.milvus_collection_name)
        self.collection = Collection(config.milvus_collection_name, self.schema)
        self.collection.create_index("embedding", self.index)  
        # 初始化向量化模型
        self.embedding = config.Embedding
        
    def load_milvus_collection(self,user_name,character_name):
        partition_name = user_name + '_'+character_name
        if self.collection.has_partition(partition_name):
            self.collection.load([partition_name], replica_number=2)
            
    def is_exist(self,user_name,character_name):
        partition_name = user_name + '_'+character_name
        if self.collection.has_partition(partition_name):
            return True
        return False

    def insert_memory(self, pk: int,  text: str,  importance_score: int,user_name:str,character_name:str):
        partition_name = user_name + '_'+character_name
        '''定义插入记忆对象函数'''
        timestamp = time.time()    
        # 使用语言模型获得文本embedding向量
        embedding = self.embedding.get_embedding_from_language_model(text)
        data = [[pk], [text], [user_name], [timestamp],[importance_score], [embedding]]
        if not self.collection.has_partition(partition_name):
            self.collection.create_partition(partition_name)
        self.collection.load_milvus_collection(user_name,character_name)
        self.collection.insert(data,partition_name)
            
    def compute_relevance(self, query_text: str, limit: int, expr: str == None,user_name:str,character_name:str):
        '''定义计算相关性分数函数'''
        # 搜索表达式
        partition_name  = user_name + '_'+character_name
        search_result = self.search(query_text, limit, expr,partition_name)
        hits = []
        for hit in search_result:
            memory = {"id": hit.entity.id,"text": hit.entity.text,"timestamp": hit.entity.timestamp,
                      "owner": hit.entity.owner,"importance_score":  hit.entity.importance_score}
            memory["relevance"] = 1 - hit.distance
            hits.append(memory)

        return hits

    def search(self, query_text: str, limit: int, expr: str == None,partition_name:str):
        query_embedding = self.embedding.get_embedding_from_language_model(query_text)
        search_params = {"metric_type": "L2", "params": {"nprobe": 30}}
        # 搜索向量关联的最新30条记忆
        vector_hits = None
        if expr != None:
            vector_hits = self.collection.search(data=[query_embedding],
                anns_field="embedding",param=search_params,limit=limit,
                expr=expr,
                output_fields=["id", "text", "owner","timestamp", "importance_score"],
                partition_name =partition_name
            )
        else:
            vector_hits = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["id", "text", "owner","timestamp", "importance_score"],
                partition_name =partition_name
            )

        return vector_hits[0]

    def compute_recency(self, memories):
        '''定义计算最近性分数函数'''
        current_time = time.time()
        for memory in memories:
            time_diff = current_time - memory["timestamp"]
            memory["recency"] = 0.99 ** (time_diff / 3600)  # 指数衰减

    def normalize_scores(self, memories):
        for memory in memories:
            memory["total_score"] = memory["relevance"] + memory["importance_score"] + memory["recency"]

    def pageQuery(self, expr: str, offset: int, limit: int,user_name:str):
        self.milvus_collection(user_name)
        vector_hits = self.collection.query(
            expr=expr,
            offset=offset,
            limit=limit,
            output_fields=["id", "text", "owner","timestamp", "importance_score"]
        )
        return vector_hits


    def release(self):
        self.collection.release()

    def clear(self, owner: str):
        ids_result = self.collection.query(
            expr=f"owner !=''",
            offset=0,
            limit=100,
            output_fields=["id"])
        ids = [item['id'] for item in ids_result]
        ids_expr = f"id in {ids}"
        self.collection.delete(ids_expr)

    # if __name__ == "__main__":

    #     # 测试代码
    #     insert_memory("John ate breakfast this morning", "John")
    #     insert_memory("Mary is planning a party for Valentine's Day", "John")
    #     insert_memory("John likes to eat BBQ", "John")
    #     insert_memory("Alan likes to eat TV", "Alan")
    #     insert_memory("Alan likes to eat Macbook", "Alan")
    #     insert_memory("John went to the library in the morning", "John")

    #     # query_text = "What are John's plans for today?"
    #     query_text = "What does Alan like?"

    #     collection.load()

    #     memories = compute_relevance(query_text, "Alan")

    #     compute_importance(memories)
    #     compute_recency(memories)
    #     normalize_scores(memories)
    #     print(memories)

    #     print("Retrieved memories:")
    #     for memory in sorted(memories, key=lambda m: m["total_score"], reverse=True)[:5]:
    #         print(memory["text"], ", total score:", memory["total_score"])

    #     # 清楚原数据
    #     utility.drop_collection("memory_stream_test11")
