"""Rag pipline--used for==function(||multiple funtions)
1.Configuration
2.Chunking  ====chunk_text
3.Vector store--chromabd ====build_vector_store
4.Ingest ===Ingest_doc||ingest_file
5.Retrival=====retrive
6.Generation with ollamma--used instead of API keys ====build_prompt||gene_ans
7.Full pipeline========= req_query
8.Interactive Chat Loop==========chat_loop
9.Main========= int __name__=="__main__"

0.1.############# DOWNLOADING SETTING UP ENVIRONMENT #################
python --m venv venv                    -----------------|
venv/Scripts/Activate                   -----------------|-----USED WHEN USING A API KEY
pip install anthropic sentence-transformers ------------|

ollama pull llama3.2                             -----------|
pip install ollama chromadb sentence-transformers ----------|------USED WHEN NO API KEY IS NEEDED TO RUN IT OFFLINE

"""

#1.#########################   CONFIGURATION     ############################
import os
import ollama
import chromadb
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
LLM_Model="llama3.2"                          #change to "phi3" if low on RAM
EMBED_Model="all-MiniLM-L6-v2"                 # runs locally via sentence-transformers
COLLECTION="rag_local"
chunck_size=400
Overlap=50
RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2" 
retrieve_k=10
rerank_top_n=3
print("loading reranker model....")
reranker=CrossEncoder(RERANK_MODEL)
NVIdia_api_key=os.environ.get("NVIDIA_API_KEY")
NVIDIA_JUDGE_LLM="meta/llama-3.1-70b-instruct"
NVIdia_JUDGE_EMBED="nvidia/nv-embedqa-e5-v5"

#2.#######################   CHUNKING       ##############################

""" USING FIXED CHUNKING
def chunk_text(text:str,size:int=chunck_size,overlap:int=Overlap)->list[str]:
    chunks=[]
    start=0
    text=text.strip()
    while start<len(text):
        end=start+size  #end=0+400 
        chunk=text[start:end] #slicing from 0 to 400
        if end<len(text):
            last_period=max(chunk.rfind("."),chunk.rfind("?"),chunk.rfind("!")) #used for proper chunking
            if last_period> size//2: #the chunck should be snaped in the second half of the size ,i.e. 400//2=200 
                end=start+last_period+1 
                chunk=text[start:end] #re-slice after last_period
        if chunk.strip():       #removing all the white spaces in the chunks
            chunks.append(chunk.strip())
        start=end-overlap  #To avoid improper meaing of chunks
    return chunks
"""
#USING RECURSIVE CHUNKING
#2.#######################   CHUNKING       ##############################

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunck_size,
    chunk_overlap=Overlap,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
)

def chunk_text(text: str) -> list[str]:
    text = text.strip()
    return splitter.split_text(text)

#3.###################### VECTOR STORE   ###############################


def build_vector_store()->chromadb.Collection:
    embed_fn=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_Model)
    client=chromadb.PersistentClient(path="./chroma_db_local")
    collection=client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space":"cosine"},
    )
    return collection


#4.#################### INGEST ###############################
def ingest_doct(collection:chromadb.Collection, text: str,source:str)->int:
    chunks=chunk_text(text)
    print(f"{source} split into {len(chunks)} chunks")
    existing=collection.count()
    ids=[f"{source}__chunk__{existing+i}"for i in range(len(chunks))]
    metadatas=[{"source":source,"chunk_index":i}for i in range(len(chunks))]
    collection.add(documents=chunks,ids=ids,metadatas=metadatas)
    print(f"->{len(chunks)} chunks stored")
    return len(chunks)
def ingest_file(collection:chromadb.Collection,filepath:str)->int:
    with open(filepath,"r",encoding="utf-8") as f:
        text=f.read()
    return ingest_doct(collection,text,source=filepath)


#5.########################### RETRIEVAL ############################


def retrieve(collection:chromadb.Collection,question:str,k:int=retrieve_k)->list[dict]:
    results=collection.query(
        query_texts=[question],
        n_results=min(k,collection.count()),
        include=["documents","metadatas","distances"],

    )
    chunks=[]
    for doc,meta,dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":doc,
            "source":meta.get("source","unknown"),
            "score":round(1-dist,3),
        })
    return chunks#

#-------------------------------------------------------------------------------------------    
#5.5###########################RERANKING##################################
def rerank(question:str,chunks:list[dict],n:int=rerank_top_n)->list[dict]:
    if not chunks:
        return chunks
    pairs=[(question,c["text"]) for c in chunks]
    scores=reranker.predict(pairs)
    for c,score in zip(chunks,scores):
        c["rerank_score"]=round(float(score),3)
    reranked=sorted(chunks,key=lambda c:c["rerank_score"],reverse=True)
    return reranked[:n]
#6.################################ GENERATION WITH OLLAMA ###########################


def build_prompt(question:str,chunks:list[dict])->str:
    if not chunks:
        return f"The user asked:'{question}'\n No relevant docts were found"
    context_parts=[]
    for i,chunk in enumerate(chunks,1):
        context_parts.append(
            f"[{i}] source:{chunk['source']} (similarity:{chunk['score']})\n {chunk['text']}"
        )
    context="\n\n".join(context_parts)
    return f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."
Cite sources as [1], [2] etc
Context:
{context}
Question:{question}
Answer:"""
def gene_ans(question:str,chunks:list[dict])->str:
    prompt=build_prompt(question,chunks)
    response=ollama.chat(
        model=LLM_Model,
        messages=[{"role":"user","content":prompt}],
    )
    return response["message"]["content"]


#7############################FULL PIPELINE #########################
def rag_query(question:str,collection:chromadb.Collection,verbose:bool=True)->str:
    chunks=retrieve(collection,question)
    if verbose:
        print(f"\n{'-'*50}")
        print(f"Question:{question}")
        print(f"\nRetrived{len(chunks)} chunks:")
        for i,c in enumerate(chunks,1):
            preview=c["text"][:100].replace("\n"," ")
            print(f"[{i}](score:{c['score']}){c['source']}-{preview}...")
        print("\nGenerating answer with ollama.............")
    chunks=rerank(question,chunks,n=rerank_top_n)
    if verbose:
        print(f"\n reranked down to top{len(chunks)}")
        for i,c in enumerate(chunks,1):
            preview=c["text"][:80].replace("\n","")
            print(f"[{i}](vector_store:{c['score']},rerank_score:{c['rerank_score']}{c['source']}-{preview})")
    answer=gene_ans(question,chunks)
    if verbose:
        print(f"\nAnswer:\n{answer}")
        print(f"{'-'*50}\n")
    return answer


#8.########################## INTERACTIVE CHAT LOOP ###########################
def  chat_loop(collection:chromadb.Collection):
    print(f"\n{'='*50}")
    print("  Local RAG Chat (powered by Ollama)")
    print("  Commands: 'quit' to exit | 'info' for stats|'eval'to run RAGAS")
    print("="*50 + "\n")
    while True:
        try:
            question=input("YOU:").strip()
        except(KeyboardInterrupt,EOFError):
            print("\nGoodbye!!!")
            break
        if not question:
            continue
        if question.lower()=="quit":
            print("\nGoodbye!!!!!!!!!!")
            break
        if question.lower()=="info":
            print(f"Collection:{collection.count()}chunks indexwd\n")
            continue
        if question.lower()=="eval":
            try:
                run_ragas_eval(collection)
            except Exception as e:
                print(f"RAGAS eval error:{e}")
            continue
        try:
            rag_query(question,collection)
        except Exception as e:
            print(f"Error:{e}")
            print("  Make sure Ollama is running: 'ollama serve' in another terminal\n")

SAMPLE_DOCS = {
    "python_basics.txt": """
Python is a high-level, interpreted programming language created by Guido van
Rossum and released in 1991. It is known for its clean, readable syntax that
uses indentation to define code blocks rather than curly braces. Python
supports multiple programming paradigms including object-oriented, functional,
and procedural styles. It is dynamically typed, meaning variable types are
determined at runtime. Python has a massive standard library and over 400,000
third-party packages on PyPI. It is widely used for web development (Django,
Flask), data science (NumPy, pandas), machine learning (TensorFlow, PyTorch),
automation, and scripting.
""",
    "rag_overview.txt": """
Retrieval-Augmented Generation (RAG) combines information retrieval with
language model generation. Documents are split into chunks, converted to
vector embeddings, and stored in a vector database like Chroma or FAISS.
When a user asks a question, it is embedded using the same model and compared
to stored chunks using cosine similarity. The most relevant chunks are
retrieved and injected into the LLM prompt as context. The LLM then generates
an answer grounded in that context. RAG reduces hallucination, keeps answers
current without retraining, and works with private documents the model has
never seen.
""",
    "ollama_info.txt": """
Ollama is a tool that lets you run large language models locally on your own
machine. It supports models like Llama, Mistral, Phi, Gemma, and many others.
Once installed, you pull a model with 'ollama pull modelname' and run it with
'ollama run modelname'. Ollama exposes a REST API on localhost:11434, and also
has a Python library for easy integration. Running models locally means no API
keys, no internet dependency, no usage costs, and full data privacy since
nothing leaves your machine.
""",
}
EVAL_QUESTIONS = [
    "When was Python created and by whom?",
    "How does RAG reduce hallucination?",
    "Can I use Ollama without internet?",
]
EVAL_GROUND_TRUTHS = [
    "Python was created by Guido van Rossum and released in 1991.",
    "RAG reduces hallucination by grounding the LLM's answer in retrieved "
    "context from real documents instead of relying only on the model's "
    "internal knowledge.",
    "Yes, Ollama runs models locally so no internet connection is needed "
    "once the model is downloaded.",
]
 
 

 
#9.################## RAGAS EVALUATION (judged via NVIDIA NIM API) ####################
def run_ragas_eval(collection:chromadb.Collection):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_nvidia_ai_endpoints import ChatNVIDIA,NVIDIAEmbeddings
    except ImportError as e:
        print(f"Missing dependecy for RAGAS eval:{e}")
        
        return
    api_key=NVIdia_api_key
    if not api_key:
        api_key=input("ENter nvida api key.........::").strip()
        if not api_key:
            print("No api_key enterted")
            return
    print(f"\n{'-'*50}")
    print("running ragas ealuation")
    print(f"Generation:ollama{LLM_Model}||judge:nvida nim{NVIDIA_JUDGE_LLM}")
    print(f"{'-'*50}")
    answers,context_list=[],[]
    for q in EVAL_QUESTIONS:
        chunks=retrieve(collection,q)
        chunks=rerank(q,chunks,n=rerank_top_n)
        answer=gene_ans(q,chunks)
        answers.append(answer)
        context_list.append([c["text"] for c in chunks])
        print(f"Question:{q}\n Answer:{answer}\n")
    dataset=Dataset.from_dict(
        {
            "question":EVAL_QUESTIONS,
            "answer":answers,
            "retrieved_contexts":context_list,
            "ground_truth":EVAL_GROUND_TRUTHS,

        }
        )
    judge_llm=LangchainLLMWrapper(
        ChatNVIDIA(model=NVIDIA_JUDGE_LLM,api_key=api_key,temperature=0.0,timeout=120)
    )
    judge_embedings=LangchainEmbeddingsWrapper(
        NVIDIAEmbeddings(model=NVIdia_JUDGE_EMBED,api_key=api_key,timeout=120)
    )
    print("Scoring with Ragas(this calls the nvidia api several times per question....)")  
    result=evaluate(
        dataset,
        metrics=[faithfulness,answer_relevancy,context_precision,context_recall],
        llm=judge_llm,
        embeddings=judge_embedings,
        raise_exceptions=False,
    )        
    print("\n==Overall ragas scores====")
    print(result)
    df=result.to_pandas()
    q_col="question" if "question" in df.columns else "user_input"
    a_col="answer" if "answer" in df.columns else "response"
    metric_cols=[c for c in ["faithfulness","answer_relevancy","context_precision","context_recall"] if c in df.columns]
    print(f"==per-question ragas scores{len(df)}questions")
    for i,row in df.iterrows():
        question_text=row[q_col]
        answer_text=str(row[a_col])
        print(f"[{i+1}]Q:{question_text}")
        print(f"A:{answer_text[:200]}{'...'if len(answer_text)>200 else " "}")
        for col in metric_cols:
            score=row[col]
            if score is None or(isinstance(score,float)and score!=score):
                print(f"{col:<20}:N/A(judge call failed/timed out)")
            else:
                print(f"{col:<20}:{score:.3f}")
        print()
    out_path="ragas_results.csv"
    df.to_csv(out_path,index=False)
    print(f"Full per_question breakdown also saved to {out_path}")


    

#10.###########################  MAIN    ########################################           

def main():
    print("Setting up RAG Systems")
    print(f"LLM MODEL USED:{LLM_Model}|| Embeding used:{EMBED_Model}\n")
    collection=build_vector_store()
    if collection.count()==0:
        print("Ingesting documents")
        for filename,content in SAMPLE_DOCS.items():
            ingest_doct(collection,content,source=filename)
        print(f"\nTotal chunks stored:{collection.count()}\n")
    else:
        print(f"Loaded existing store:{collection.count()} chunks\n")
    demo_questions = [
        "How many library are there in python",
        "How does RAG reduce hallucination?",
        "Can I use Ollama without internet?",
    ]
    print(">>>>>> Running demo question >>>>>>>>>>>")
    for q in demo_questions:
        rag_query(q,collection,verbose=True)
    
    chat_loop(collection)
if __name__=="__main__":
    main()