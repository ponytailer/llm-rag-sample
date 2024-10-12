from dspy.predict.langchain import LangChainModule, LangChainPredict
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM


set_llm_cache(SQLiteCache("langchain_cache.db"))
llm = OllamaLLM(model="llama3", stream=False)
retriever = WikipediaRetriever(load_max_docs=1)
prompt = PromptTemplate.from_template(
        "Given{context},answer the question `{question}` as a tweet.")


def retrieve(inputs):
    return [doc.page_content[:1024] for doc in
            retriever.get_relevant_documents(query=inputs["question"])]


question = "where was MS Dhoni born?"
module = LangChainModule(
        RunnablePassthrough.assign(context=retrieve))
predict = LangChainPredict(prompt, llm)
zeroshot_chain = module | predict | StrOutputParser()
