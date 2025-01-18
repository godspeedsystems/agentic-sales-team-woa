import os
import dotenv
import asyncio
from datetime import datetime
dotenv.load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.agent.react import ReActAgent
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core import Settings

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
Settings.embed_model = embedding_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")
Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)

# Event classes remain unchanged
class JudgeEvent(Event):
    query: str

class BadQueryEvent(Event):
    query: str

class NaiveRAGEvent(Event):
    query: str

class HighTopKEvent(Event):
    query: str

class RerankEvent(Event):
    query: str

class ResponseEvent(Event):
    query: str
    response: str
    source: str  # Added source field to track which strategy generated the response

class SummarizeEvent(Event):
    query: str
    response: str

class ComplicatedWorkflow(Workflow):
    def load_or_create_index(self, directory_path, persist_dir):
        embedding_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")
        
        if os.path.exists(persist_dir):
            print("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
        else:
            print("Creating new index...")
            documents = SimpleDirectoryReader(directory_path).load_data()
            
            # Create index with Gemini embedding model
            index = VectorStoreIndex.from_documents(
                documents, 
                embed_model=embedding_model
            )
            
            index.storage_context.persist(persist_dir=persist_dir)
        
        return index

    @step(pass_context=True)
    async def judge_query(self, ctx: Context, ev: StartEvent | JudgeEvent ) -> BadQueryEvent | NaiveRAGEvent | HighTopKEvent | RerankEvent:
        if not hasattr(ctx.data, "llm"):
            llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
            ctx.data["llm"] = llm
            ctx.data["index"] = self.load_or_create_index(
                "data_short",
                "storage"
            )
            # Using Gemini for chat engine
            ctx.data["judge"] = SimpleChatEngine.from_defaults(
                llm=llm,
                embed_model=GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")
            )

        response = ctx.data["judge"].chat(f"""
            Given a user query, determine if this is likely to yield good results from a RAG system as-is. If it's good, return 'good', if it's bad, return 'bad'.
            Good queries use a lot of relevant keywords and are detailed. Bad queries are vague or ambiguous.

            Here is the query: {ev.query}
            """)
        if response == "bad":
            return BadQueryEvent(query=ev.query)
        else:
            self.send_event(NaiveRAGEvent(query=ev.query))
            self.send_event(HighTopKEvent(query=ev.query))
            self.send_event(RerankEvent(query=ev.query))

    @step(pass_context=True)
    async def improve_query(self, ctx: Context, ev: BadQueryEvent) -> JudgeEvent:
        response = ctx.data["llm"].complete(f"""
            This is a query to a RAG system: {ev.query}

            The query is bad because it is too vague. Please provide a more detailed query that includes specific keywords and removes any ambiguity.
        """)
        return JudgeEvent(query=str(response))

    @step(pass_context=True)
    async def naive_rag(self, ctx: Context, ev: NaiveRAGEvent) -> ResponseEvent:
        index = ctx.data["index"]
        engine = index.as_query_engine(
            similarity_top_k=5,
            embed_model=GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")
        )
        response = engine.query(ev.query)
        print("Naive response:", response)
        return ResponseEvent(query=ev.query, source="Naive", response=str(response))

    @step(pass_context=True)
    async def high_top_k(self, ctx: Context, ev: HighTopKEvent) -> ResponseEvent:
        index = ctx.data["index"]
        engine = index.as_query_engine(
            similarity_top_k=20,
            embed_model=GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")
        )
        response = engine.query(ev.query)
        print("High top k response:", response)
        return ResponseEvent(query=ev.query, source="High top k", response=str(response))

    @step(pass_context=True)
    async def rerank(self, ctx: Context, ev: RerankEvent) -> ResponseEvent:
        index = ctx.data["index"]
        reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=5)
        
        embedding_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")
        
        retriever = index.as_retriever(
            similarity_top_k=20,
            embed_model=embedding_model
        )
        
        engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
        )

        response = engine.query(ev.query)
        print("Reranker response:", response)
        return ResponseEvent(query=ev.query, source="Reranker", response=str(response))

    @step(pass_context=True)
    async def judge(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        ready = ctx.collect_events(ev, [ResponseEvent]*3)
        if ready is None:
            return None

        response = ctx.data["judge"].chat(f"""
            A user has provided a query and 3 different strategies have been used
            to try to answer the query. Your job is to decide which strategy best
            answered the query. The query was: {ev.query}

            Response 1 ({ready[0].source}): {ready[0].response}
            Response 2 ({ready[1].source}): {ready[1].response}
            Response 3 ({ready[2].source}): {ready[2].response}

            Please provide the number of the best response (1, 2, or 3).
            Just provide the number, with no other text or preamble.
        """)

        best_response = int(str(response))
        print(f"Best response was number {best_response}, which was from {ready[best_response-1].source}")
        return StopEvent(result=str(ready[best_response-1].response))

async def main():
    c = ComplicatedWorkflow(timeout=60, verbose=True)

    result = await c.run(
        query="What are the different types of overloads?"
    )
    print(result)

if __name__ == "__main__":
    start = datetime.now()
    asyncio.run(main())
    print("\n Time Taken = ", datetime.now()-start)