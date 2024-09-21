from haystack.components.builders import PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.core.pipeline import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.ollama import \
    OllamaDocumentEmbedder, OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator

document_store = InMemoryDocumentStore()
# 输入
text_file_converter = TextFileToDocument()
# 清洗: 组件的任务是提升文本文档的可读性，你可以通过它删除空行、多余空格或重复的子字符串等。
cleaner = DocumentCleaner()
# 分块
splitter = DocumentSplitter(split_by="sentence", split_length=5)

embedder = OllamaDocumentEmbedder()

# 它是第一个管道的输出部分，负责将文档列表写入最初定义的文档存储中，以便在下一个管道中使用
writer = DocumentWriter(document_store)

retriever = InMemoryEmbeddingRetriever(document_store)


def create_pipeline():
    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component("converter", text_file_converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline \
        .connect("converter.documents", "cleaner.documents") \
        .connect("cleaner.documents", "splitter.documents") \
        .connect("splitter.documents", "embedder.documents") \
        .connect("embedder.documents", "writer.documents")

    return indexing_pipeline


template = """根据相关文档，给出答案
Question:{{query}}
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
Answer:"""
prompt_builder = PromptBuilder(template=template)
llm = OllamaGenerator(model="llama3")

if __name__ == "__main__":
    indexing_pipeline = create_pipeline()
    indexing_pipeline.run(data={"sources": ["davinci.txt"]})
    # indexing_pipeline.draw(Path("./haystack.png"))

    rag_pipeline = Pipeline()

    text_embedder = OllamaTextEmbedder()

    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    # rag_pipeline.draw(Path("./haystack-rag.png"))

    query = "虚引用是什么"

    result = rag_pipeline.run(
        data={
            "prompt_builder": {"query": query},
            "text_embedder": {"text": query}
        })
    print(result["llm"]["replies"][0])
