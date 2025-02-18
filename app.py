import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA, NVIDIARerank
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from typing import Union, List
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import Faithfulness,LLMContextRecall,FactualCorrectness
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# langsmith_key = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_API_KEY"] =st.secrets("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]= "Chat-with-doc"
   
if langsmith_key:
    print("Langsmith API key loaded successfully!")
else:
    print("Langsmith API key not found. Please check your .env file.")

OPENAI_MODELS = {
    "chat": "gpt-3.5-turbo",
    "embeddings": "text-embedding-3-small"
}

# API Keys
OPENAI_API_KEY = st.secrets("OPENAI_API_KEY")
NVIDIA_API_KEY = st.secrets("NVIDIA_API_KEY")
GEMINI_API_KEY = st.secrets("GEMINI_API_KEY")

sample_queries = [
                "What is Transformer?",
                "What is attention?",
                "What is Self-Attention?",
                "What is multi head attention?",
                "What is a Large Language Model (LLM)?",
                "What is RAG (Retrieval-Augmented Generation)?",
                "What is LangChain?"
            ]
expected_responses = [
    """A Transformer is a type of neural network architecture introduced in the 2017 paper “Attention Is All You Need” by Vaswani et al.
        It has become the backbone for many state-of-the-art natural language processing models.  
        Here are the key points about Transformers:
        Architecture: Unlike recurrent neural networks (RNNs), which process input sequences sequentially,
        transformers handle input sequences in parallel via a self-attention mechanism.
        Key components: Encoder-Decoder structure, Multi-head attention layers, Feed-forward neural networks,
        Positional encodings
        Self-attention: This feature enables the model to efficiently capture long-range relationships by assessing
        the relative relevance of various input components as it processes each element.
        parallelisation: Transformers can handle all input tokens concurrently, which speeds up training and
        inference times compared to RNNs.
        Scalability: Transformers can handle longer sequences and larger datasets more effectively than previous
        architectures.
        Versatility: Transformers were first created for machine translation, but they have now been modified for
        various NLP tasks, including computer vision applications.
        Impact: Transformer-based models, including BERT, GPT, and T5, are the basis for many generative AI applications and have broken records in various language tasks.
        Transformers have revolutionized NLP and continue to be crucial components in the development of advanced AI models.""",
 
    """Attention is a technique used in generative AI and neural networks that allows models to focus on specific
        input areas when generating output. It enables the model to dynamically ascertain the relative importance
        of each input component in the sequence instead of considering all the input components similarly. """,
 
    """ Also referred to as intra-attention, self-attention enables a model to focus on various points within an
        input sequence. It plays a crucial role in transformer architectures.
        How does it work?
        Three vectors are created for each element in a sequence: query (Q), Key (K), and Value (V).
        Attention scores are computed by taking the dot product of the Query with all Key vectors.
        These scores are normalized using softmax to get attention weights.
        The final output is a weighted sum of the Value vectors, using the attention weights.
        Benefits:
        Captures long-range dependencies in sequences.
        Allows parallel computation, making it faster than recurrent methods.
        Provides interpretability through attention weights.""",
 
    """This technique enables the model to attend to data from many representation subspaces by executing numerous
        attention processes simultaneously.
        How does it work?
        The input is linearly projected into multiple Query, Key, and Value vector sets.
        Self-attention is performed on each set independently.
        The results are concatenated and linearly transformed to produce the final output.
        Benefits:
        Allows the model to jointly attend to information from different perspectives.
        Improves the representation power of the model.
        Stabilizes the learning process of attention mechanisms.""",
 
    """A large language model (LLM) is a type of artificial intelligence (AI) program that can recognize and
        generate text, among other tasks. LLMs are trained on huge sets of data — hence the name “large.”
        LLMs are built on machine learning; specifically, a type of neural network called a transformer model.""",
 
    """Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model,
        so it references an authoritative knowledge base outside of its training data sources before generating
        a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters
        to generate original output for tasks like answering questions, translating languages, and completing
        sentences. RAG extends the already powerful capabilities of LLMs to specific domains or an organization’s
        internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to
        improving LLM output so it remains relevant, accurate, and useful in various contexts.""",
    """An open-source framework called LangChain creates applications based on large language models (LLMs).
        Large deep learning models known as LLMs are pre-trained on vast amounts of data and can produce answers
        to user requests, such as generating images from text-based prompts or providing answers to enquiries.
        To increase the relevance, accuracy, and degree of customisation of the data produced by the models,
        LangChain offers abstractions and tools. For instance, developers can create new prompt chains or alter
        pre-existing templates using LangChain components. Additionally, LangChain has parts that let LLMs use
        fresh data sets without having to retrain."""
  ]
class OpenAIHandler:
    def __init__(self, openai_api_key: str):
        self.chat_client = ChatOpenAI(
            model=OPENAI_MODELS["chat"],
            api_key=openai_api_key,
            temperature=0,
            max_tokens=4000
        )
        self.embedding_client =OpenAIEmbeddings(model=OPENAI_MODELS["embeddings"], api_key=openai_api_key)

    def generate_summary(self, text: str) -> str:
        start_time = time.time()
        prompt = f"""
            You are an expert summarizer. Summarize the following text clearly and concisely, ensuring coherence and readability.
            Focus on capturing key points, important facts, and main ideas while avoiding unnecessary details.
   
            - Use simple and clear language.
            - Maintain the original meaning while making it brief.
   
            Text to summarize:
            \"\"\"{text}\"\"\"
   
            Provide a well-structured summary in a paragraph format.
        """
        response = self.chat_client.invoke(prompt)
        end_time = time.time()
        return response.content, end_time - start_time


    def query_document(self, query: str, context: str) -> str:
        start_time = time.time()
        """Query the document with the provided context."""
        prompt = f"""
            You are an intelligent AI assistant. Answer the following query based on the given passage.
            If the passage does not contain enough information, say **"The passage does not provide enough details to answer this question."**
            Do not make up any information.
 
            ---
            **Passage:**
            {context}
           
            **Query:**
            {query}
 
            **Answer:**
        """
        response = self.chat_client.invoke(prompt)
        end_time = time.time()
        return response.content, end_time - start_time


class NVIDIAHandler:
    def __init__(self, nvidia_api_key: str):
        self.chat_client = ChatNVIDIA(
            #model="meta/llama-3.1-70b-instruct",
            model="nvidia/nemotron-4-340b-instruct",
            api_key=nvidia_api_key,
            temperature=0,
            max_tokens=4000
        )
        self.embedding_client =NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=nvidia_api_key)
        self.rerank_client = NVIDIARerank(
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
            api_key=nvidia_api_key,
        )

    def generate_summary(self, text: str) -> str:
        start_time = time.time()
        prompt = f"""
            You are an expert summarizer. Summarize the following text clearly and concisely, ensuring coherence and readability.
            Focus on capturing key points, important facts, and main ideas while avoiding unnecessary details.
   
            - Use simple and clear language.
            - Maintain the original meaning while making it brief.
   
            Text to summarize:
            \"\"\"{text}\"\"\"
   
            Provide a well-structured summary in a paragraph format.
        """
        response = self.chat_client.stream([{"role": "user", "content": prompt}])
        summary = []
        for chunk in response:  # Process streaming response
            summary.append(chunk.content)
        end_time = time.time()
        return  "".join(summary), end_time - start_time


    def query_document(self, query: str, context: str) -> str:
        start_time = time.time()
        """Query the document with the provided context."""
        prompt = f"""
            You are an intelligent AI assistant. Answer the following query based on the given passage.
            If the passage does not contain enough information, say **"The passage does not provide enough details to answer this question."**
            Do not make up any information.
 
            ---
            **Passage:**
            {context}
           
            **Query:**
            {query}
 
            **Answer:**
        """
        response = self.chat_client.stream([{"role": "user", "content": prompt}])
        end_time = time.time()
        return "".join(chunk.content for chunk in response), end_time - start_time

class ModelEvaluation:
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=gemini_api_key,  # Explicitly pass the API key
            temperature=0,
            convert_system_message_to_human=True
        )    
    def get_gemini_evaluation_summary(self, openai_summary: str, nvidia_summary: str, document: str):
        """Get evaluation summary from Gemini for summarization task."""
        from langchain_core.messages import HumanMessage
        
        try:
            prompt = f"""
        You are an expert evaluator assessing the quality of summarization outputs.
        Compare the following two summaries generated by OpenAI and NVIDIA LLMs based on the original document.
 
        **Evaluation Criteria (Score out of 10 for each summary, ensuring that the better summary receives a higher score where justified):**
        - Assign higher scores to summaries that are **more structured, well-formatted, and clearly articulated**.
        - If one summary provides **better organization with bullet points or sections**, it must be rewarded.
        - If both summaries are similar in quality, provide a balanced score.
        - Ensure that a structured, well-formatted summary with clear pointers is **not penalized unfairly**.
 
        Create a comparison matrix with two columns (**OpenAI, NVIDIA**) and three rows (**Accuracy, Coherence, Correctness**):
 
        1. **Accuracy** - How well does the summary capture the key points of the original document?
        - Reward summaries that **cover all critical aspects concisely and completely**.
        - Penalize summaries that miss key points or introduce irrelevant details.
 
        2. **Coherence** - How logically and fluently is the summary structured?
        - Higher scores for summaries that are **well-organized, easy to follow, and use bullet points or a structured format**.
 
        3. **Correctness** - Are there any factual inaccuracies, misinterpretations, or hallucinations?
        - Deduct points for factual errors, contradictions, or misinterpretations.
 
        **Task:**
        - Provide a **structured comparison table** with OpenAI and NVIDIA as columns and Accuracy, Coherence, Correctness as rows.
       
        - Identify **which summary is better overall** based on the total score.
 
        **Original Document Context:**
        {document}
 
        **OpenAI Summary:**
        {openai_summary}
 
        **NVIDIA Summary:**
        {nvidia_summary}
        """
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error in Gemini evaluation: {str(e)}"

    def get_gemini_evaluation_qna(self, openai_response: str, nvidia_response: str, document: str, query: str, task="question-answering"): 
        """Get evaluation from Gemini for QnA task."""
        from langchain_core.messages import HumanMessage
        
        try:
            # prompt = f"""
            # Compare the following two outputs for {task}. Input question is based on the {query} and {document}. Provide a score out of 10 for accuracy, coherence, and correctness.
            # **Create table with two column named openAI and NVIDIA and three rows named accuracy, coherence, and correctness.**
            # Also explain which one is better and why.
            
            # OpenAI Output:
            # {openai_response}
        
            # NVIDIA Output:
            # {nvidia_response}
            # """
            # prompt = f""" 
            # Compare the following two outputs for the task: {task}.  The input question is based on the query: "{query}" and the document: "{document}".  
            # ### Evaluation Criteria:   
            # Rate each response on a scale of 1 to 10 for:   
            # - **Accuracy** (How well the response aligns with the correct answer)   
            # - **Coherence** (How well-structured and understandable the response is)   
            # - **Correctness** (Whether the response contains factual errors)   
            # Task:
            # - Provide a **structured comparison table** with OpenAI and NVIDIA as columns and Accuracy, Coherence, Correctness as rows.
            # ### Scoring Table:   
            # Present the scores in a properly formatted table with the following structure:
            # | Criteria   | OpenAI | NVIDIA |
            # |------------|--------|--------| 
            # | Accuracy   | X      | Y      | 
            # | Coherence  | X      | Y      | 
            # | Correctness| X      | Y      |

            # ### Comparative Analysis:   
            # - Which model performed better overall?   
            # - Justify your evaluation with specific strengths and weaknesses of each response.  
            
            # #### OpenAI Output:
            # {openai_response} 
            # #### NVIDIA Output: 
            # {nvidia_response} 
            # """
            
            prompt = f"""
        You are an expert evaluator assessing the quality of response outputs.
        Compare the following two outputs for {task}. Input question is based on the {query} and {document}.
        
        **Evaluation Criteria (Score out of 10 for each response, ensuring that the better response receives a higher score where justified):**
        - Assign higher scores to response that is **more structured, well-formatted, and clearly articulated**.
        
        - If both response are similar in quality, provide a balanced score.
 
        Create a comparison matrix with two columns (**OpenAI, NVIDIA**) and three rows (**Accuracy, Coherence, Correctness**):
 
        1. **Accuracy** - How well does the response capture the key points of the original document?
 
        2. **Coherence** - How logically and fluently is the response structured?
 
        3. **Correctness** - Are there any factual inaccuracies, misinterpretations, or hallucinations?
 
        **Task:**
        - Provide a **structured comparison table** with OpenAI and NVIDIA as columns and Accuracy, Coherence, Correctness as rows.
       
        - Identify **which response is better overall for ** based on the total score.
 
        **Original Document Context:**
        {document}
 
        **OpenAI Summary:**
        {openai_response}
 
        **NVIDIA Summary:**
        {nvidia_response}
        """
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error in Gemini evaluation: {str(e)}"
   

def process_pdf(uploaded_file):
    """Process the uploaded PDF file and extract text."""
    reader = PdfReader(uploaded_file)
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)



def query_rag_both_openai(handler,documents,query):
    start_time = time.time()
    """Retrieve relevant document chunks and generate a response using the selected LLM."""
    doc_objects_openai= [Document(page_content=text) for text in documents]
    vector_store_openai=FAISS.from_documents(doc_objects_openai,embedding=OpenAIEmbeddings(model=OPENAI_MODELS["embeddings"], api_key=OPENAI_API_KEY))  
    vector_store_openai.save_local("faiss_index_openai")
 
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retrieved_docs_openai = vector_store_openai.similarity_search(query, k=3) 
    retriever_openai = vector_store_openai.as_retriever(passages=retrieved_docs_openai[0].page_content, search_kwargs={"k": 5}) 
    st.session_state.retriever_openai = retriever_openai  
    chain_openai = ConversationalRetrievalChain.from_llm(
        llm=handler.chat_client,
        retriever=retriever_openai,
        memory=st.session_state.memory
    )
    openai_response=chain_openai.invoke(query)
    end_time = time.time()
    return openai_response["answer"], end_time - start_time

def query_rag_both_nvidia(handler,documents,query):
    
    start_time = time.time()
    """Retrieve relevant document chunks and generate a response using the selected LLM."""
    doc_objects_nvidia = [Document(page_content=text) for text in documents]
    vector_store_nvidia=FAISS.from_documents(doc_objects_nvidia,embedding=NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=NVIDIA_API_KEY))  
    vector_store_nvidia.save_local("faiss_index_nvidia")
    retrieved_docs_nvidia = vector_store_nvidia.similarity_search(query, k=3)    
    reranked_docs = handler.rerank_client.compress_documents(query=query, documents=retrieved_docs_nvidia)
    st.session_state.reranked_docs = reranked_docs
    top_passage = reranked_docs[0].page_content
    retriever_nvidia=vector_store_nvidia.as_retriever(passages=top_passage, search_kwargs={"k": 5})
    chain_nvidia =ConversationalRetrievalChain.from_llm(
        llm=handler.chat_client,
        retriever=retriever_nvidia  ,
        memory=st.session_state.memory
    )
    nvidia_response=chain_nvidia.invoke(query)
    end_time = time.time()
    return nvidia_response["answer"], end_time - start_time

def query_rag(handler, documents: List[str], query: str,model_choice=None, rerank: bool = True) -> str:
    start_time = time.time()
    """Retrieve relevant document chunks and generate a response using the selected LLM."""
    doc_objects = [Document(page_content=text) for text in documents]
    if model_choice == "OpenAI":
       vector_store=FAISS.from_documents(doc_objects,embedding=OpenAIEmbeddings(model=OPENAI_MODELS["embeddings"], api_key=OPENAI_API_KEY)) 
    elif model_choice == "NVIDIA":
        vector_store=FAISS.from_documents(doc_objects,embedding=NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=NVIDIA_API_KEY))              
   
    vector_store.save_local("faiss_index")
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retrieved_docs = vector_store.similarity_search(query, k=3)
    st.session_state.retriever_openai=retrieved_docs
    if rerank and isinstance(handler, NVIDIAHandler):
        reranked_docs = handler.rerank_client.compress_documents(query=query, documents=retrieved_docs)
        st.session_state.reranked_docs = reranked_docs
        top_passage = reranked_docs[0].page_content
    else:
        top_passage = retrieved_docs[0].page_content

    retriever = vector_store.as_retriever(passages=top_passage, search_kwargs={"k": 5})

    chain = ConversationalRetrievalChain.from_llm(
        llm=handler.chat_client,
        retriever=retriever,
        memory=st.session_state.memory
    )
    
    response=chain.invoke(query)
    end_time = time.time()
    return response["answer"], end_time - start_time

# Initialize session state variables
import streamlit as st

import time

def initialize_session_state():
    """Initialize all session state variables while ensuring UI elements persist"""
    default_states = {
        "openai_summary": "",
        "nvidia_summary": "",
        "openai_response": "",
        "nvidia_response": "",
        "comparison_result_summary": "",
        "comparison_result_response": "",
        "current_query": "",  # Preserve user query
        "active_tab": 0,  # Ensure active tab is retained
        "openai_summary_time": 0.0,
        "nvidia_summary_time": 0.0,
        "openai_query_time": 0.0,
        "nvidia_query_time": 0.0,
        "query_processing": False,
        "summary_processing": False,
        "retriever_openai": "",
        "reranked_docs": "",
        "query": "",
        "ragas_evaluation_result": ""
    }
    
    for key, default in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_generated_content():
    """Clear only generated content while preserving document and UI state."""
    keys_to_clear = [
        "openai_summary", "nvidia_summary",
        "openai_response", "nvidia_response",
        "comparison_result_summary", "comparison_result_response",
        "openai_summary_time", "nvidia_summary_time",
        "openai_query_time", "nvidia_query_time",
        "query_processing", "summary_processing", "current_query",
        "retriever_openai", "reranked_docs", "query" ,"ragas_evaluation_result"
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]  # ✅ Properly remove keys from session state

    st.session_state.query_processing = False  # Ensure processing flags are reset
    st.session_state.summary_processing = False

    st.rerun()  # Fixing `None` syntax error

                    
def set_page_config():
    """Configure the page settings and style"""
    st.set_page_config(
        page_title="RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        .warning-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            margin: 1rem 0;
        }
        .response-box {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-container {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Configure and display sidebar elements"""
    with st.sidebar:
        st.image("https://tse3.mm.bing.net/th/id/OIP.SLIBlXEhyee_i1s4PiSESQHaEL?w=545&h=307&rs=1&pid=ImgDetMain", width=250)
        st.markdown("---")
        
        model_choice = st.selectbox(
            "Choose LLM Service",
            ["OpenAI", "NVIDIA", "Both"],
            help="Select which model(s) to use for analysis"
        )
        
        if st.button("Clear Generated Content", type="secondary"):
           clear_generated_content()

        
        st.markdown("---")
    
    return model_choice
def show_system_metrics(model_choice):
    """Displays response times in the sidebar based on model choice."""
    st.sidebar.subheader("📊 Response Times")

    # Summary Response Times
    # if st.session_state.summary_processing:
    st.sidebar.write("**Summary Generation Time:**")
    if model_choice == "OpenAI" and st.session_state.openai_summary_time is not None:
        st.sidebar.metric("OpenAI Summary Time", f"{st.session_state.openai_summary_time:.2f}s")
    elif model_choice == "NVIDIA" and st.session_state.nvidia_summary_time is not None:
        st.sidebar.metric("NVIDIA Summary Time", f"{st.session_state.nvidia_summary_time:.2f}s")
    elif model_choice == "Both":
        if st.session_state.openai_summary_time is not None:
            st.sidebar.metric("OpenAI Summary Time", f"{st.session_state.openai_summary_time:.2f}s")
        if st.session_state.nvidia_summary_time is not None:
            st.sidebar.metric("NVIDIA Summary Time", f"{st.session_state.nvidia_summary_time:.2f}s")

    # Query Response Times
    # if st.session_state.query_processing:
    st.sidebar.write("**Query Response Time:**")
    if model_choice == "OpenAI" and st.session_state.openai_query_time is not None:
        st.sidebar.metric("OpenAI Query Time", f"{st.session_state.openai_query_time:.2f}s")
    elif model_choice == "NVIDIA" and st.session_state.nvidia_query_time is not None:
        st.sidebar.metric("NVIDIA Query Time", f"{st.session_state.nvidia_query_time:.2f}s")
    elif model_choice == "Both":
        if st.session_state.openai_query_time is not None:
            st.sidebar.metric("OpenAI Query Time", f"{st.session_state.openai_query_time:.2f}s")
        if st.session_state.nvidia_query_time is not None:
            st.sidebar.metric("NVIDIA Query Time", f"{st.session_state.nvidia_query_time:.2f}s")

def show_progress():
    """Display a progress bar for operations"""
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    return progress_bar

def process_document(handler, text, query=None, is_summary=True, model_choice=None):
    """Process document with progress indication and RAG retrieval"""
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)
        progress_text.text(f"{'Generating Summary' if is_summary else 'Processing Query'}: {i+1}%")

    result, response_time = None, 0

    if is_summary:
        result, response_time = handler.generate_summary(text)
    
    elif query is not None:
        # ✅ Check if handler is OpenAIHandler or NVIDIAHandler
        if model_choice == "Both":
            if isinstance(handler, OpenAIHandler):
                result, response_time = query_rag_both_openai(handler, text, query)
            elif isinstance(handler, NVIDIAHandler):
                result, response_time = query_rag_both_nvidia(handler, text, query)
        else:
            result, response_time = query_rag(handler, text, query, model_choice)

    progress_text.empty()
    progress_bar.empty()
    
    return result, response_time



def handle_single_model_query(model_choice, documents, query):
    """Handles query processing for a single model."""
    
    handler = OpenAIHandler(OPENAI_API_KEY) if model_choice == "OpenAI" else NVIDIAHandler(NVIDIA_API_KEY)

    with st.spinner(f"Processing query with {model_choice}..."):
        if model_choice== "OpenAI":
          openai_response, openai_query_time = process_document(handler, documents, query, is_summary=False, model_choice=model_choice)
          st.session_state.openai_response=openai_response
          st.session_state.openai_query_time=openai_query_time
        else:
          nvidia_response, nvidia_query_time =process_document(handler,documents, query, is_summary=False, model_choice=model_choice) 
          st.session_state.nvidia_response=nvidia_response
          st.session_state.nvidia_query_time=nvidia_query_time
        

        # ✅ Store response in session state (but don't display it here)
        # st.session_state[f"{model_choice.lower()}_response"] = openai_response if model_choice == "OpenAI" else nvidia_response
        # st.session_state[f"{model_choice.lower()}_query_time"] = openai_query_time if model_choice == "OpenAI" else nvidia_query_time

        
def handle_both_models_query(documents, query):
    """Handles query processing for both OpenAI and NVIDIA models."""

    model_choice = "Both"

    with st.spinner("Processing query with OpenAI..."):
        openai_handler = OpenAIHandler(OPENAI_API_KEY)
        openai_response, openai_query_time = process_document(openai_handler, documents, query, is_summary=False, model_choice=model_choice)

        # ✅ Store OpenAI response but do not display
        st.session_state.openai_response = openai_response
        st.session_state.openai_query_time = openai_query_time

    with st.spinner("Processing query with NVIDIA..."):
        nvidia_handler = NVIDIAHandler(NVIDIA_API_KEY)
        nvidia_response, nvidia_query_time = process_document(nvidia_handler, documents, query, is_summary=False, model_choice=model_choice)

        # ✅ Store NVIDIA response but do not display
        st.session_state.nvidia_response = nvidia_response
        st.session_state.nvidia_query_time = nvidia_query_time

def ragas_evaluation_tab(query, model_choice, single_model):
    """Handles the UI and logic for the RAGAS evaluation tab."""
   
    st.session_state.active_tab = 3
    colored_header("Ragas Evaluation", description="Evaluate model outputs", color_name="blue-70")
    if st.button("ragas evaluation", type="primary"):
        if st.session_state.openai_response or st.session_state.nvidia_response:
            if single_model:
                process_ragas_evaluation_single(query, model_choice)
            elif model_choice == "Both":
                process_ragas_evaluation_both(query)
        else:
            st.warning("Please generate summaries and responses before proceeding with the RAGAS evaluation.")        
            

def process_ragas_evaluation_single(query, model_choice):
    """Processes RAGAS evaluation for a single model (OpenAI or NVIDIA)."""
    # handler = OpenAIHandler(OPENAI_API_KEY).chat_client if model_choice == "OpenAI" else NVIDIAHandler(NVIDIA_API_KEY).chat_client
    handler=ModelEvaluation(GEMINI_API_KEY).llm
    print(query)
    if st.session_state.openai_response or st.session_state.nvidia_response:
        with st.spinner("Generating Ragas comparison..."):
            evaluate_single_model(query, model_choice, handler)

def process_ragas_evaluation_both(query):
    """Processes RAGAS evaluation for both models (OpenAI & NVIDIA)."""
    if st.session_state.openai_response and st.session_state.nvidia_response:
        with st.spinner("Generating Ragas comparison for both models..."):
            evaluate_both_models(query)

def evaluate_single_model(query, model_choice, llm):
    """Evaluates RAGAS metrics for a single model."""
    dataset = []
    query1=query
    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = st.session_state.retriever_openai if model_choice == "OpenAI" else st.session_state.reranked_docs
        response = st.session_state.openai_response if model_choice == "OpenAI" else st.session_state.nvidia_response
        context_texts = [doc[0] if isinstance(doc, tuple) else doc.page_content for doc in relevant_docs]
        if query==query1:
            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": context_texts ,
                    "response": response,
                    "reference": reference
                }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(llm)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm
    )

    display_evaluation_results(result)

def evaluate_both_models(query):
    """Evaluates RAGAS metrics for both OpenAI and NVIDIA models."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OpenAI Ragas Score")
        evaluate_single_model(query, "OpenAI",ModelEvaluation(GEMINI_API_KEY).llm)
    
    with col2:
        st.subheader("NVIDIA Ragas Score")
        evaluate_single_model(query, "NVIDIA",ModelEvaluation(GEMINI_API_KEY).llm)

def display_evaluation_results(result):
    """Displays the evaluation results in Streamlit."""
    result_df = result.to_pandas()
    
    # Debug: Print the actual column names
    print("Available columns in result_df:", result_df.columns.tolist())
    
    # Updated metrics mapping with possible variations of column names
    metrics_to_show = {
        'faithfulness': ['faithfulness', 'Faithfulness'],
        'context_recall': ['context_recall', 'llm_context_recall', 'LLMContextRecall'],
        'factual_correctness': ['factual_correctness', 'FactualCorrectness']
    }
    
    filtered_results = {}
    for metric_display, possible_names in metrics_to_show.items():
        # Try each possible column name
        for col_name in possible_names:
            if col_name in result_df.columns:
                metric_value = result_df[col_name].iloc[0]
                if pd.notna(metric_value):
                    filtered_results[metric_display.replace('_', ' ').title()] = [round(float(metric_value), 3)]
                else:
                    filtered_results[metric_display.replace('_', ' ').title()] = ["N/A"]
                break  # Found the column, no need to check other possible names
    
    if filtered_results:
        display_df = pd.DataFrame(filtered_results).T
        display_df.columns = ['Score']
        st.write("**RAGAS Evaluation Metrics:**")
        st.session_state.ragas_evaluation_result = display_df
        st.table(display_df)
    else:
        st.warning("No valid evaluation metrics available.")
    
def main():
    initialize_session_state()
    set_page_config()
    
    colored_header(
        label="RAG System with NVIDIA and OpenAI Integration",
        description="Upload PDF documents for analysis and comparison",
        color_name="green-70"
    )
    st.session_state.query_procesing=False
    st.session_state.summary_processing=False
    
    model_choice = display_sidebar()
    
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type="pdf",
        help="Upload your PDF document for analysis"
    )
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file  # Store uploaded file
        if "documents" not in st.session_state:
            st.session_state.documents = process_pdf(uploaded_file)

    if "uploaded_file" in st.session_state and st.session_state.uploaded_file:
        documents = st.session_state.documents
        single_model = model_choice in ["OpenAI", "NVIDIA"]
        both_models = model_choice == "Both"

        tabs = st.tabs([
            "📝 Generate Summary",
            "❓ Query PDF Content",
            "🔄 Compare Responses",
            "📊 RAGAS Evaluation"
        ])
        # if model_choice=="Both":
        #     st.session_state.openai_response=None
        #     st.session_state.nvidia_response=None
        #     st.session_state.openai_summary=None
        #     st.session_state.nvidia_summary=None
        #     st.session_state.openai_summary_time=0.0
        #     st.session_state.nvidia_summary_time=0.0
        #     st.session_state.openai_query_time=0.0
        #     st.session_state.nvidia_query_time=0.0

        # with tabs[st.session_state.active_tab]:
        with tabs[0]:
            st.session_state.active_tab = 0
            colored_header("Document Summary", description="Generate and view document summaries", color_name="blue-70")
            
            if single_model:
                handle_single_model_summary(model_choice, documents)
            else:
                handle_both_models_summary(documents)

        with tabs[1]:
            st.session_state.active_tab = 1
            colored_header("Query Document", description="Ask questions about your document", color_name="blue-70")
            
            query = st.text_input("Enter your query:", placeholder="What would you like to know about the document?")
            
            handle_query_processing(query, model_choice, documents, single_model, both_models)
            
        
            
        if model_choice == "Both":
            with tabs[2]:
                st.session_state.active_tab = 2
                colored_header("Response Comparison", description="Compare model outputs", color_name="blue-70")
                handle_comparisons(documents, query) 
        else:
            with tabs[2]:
                st.session_state.active_tab = 2
                if model_choice =="OpenAI" or model_choice=="NVIDIA":
                    st.warning("compare responses  only when you select the model choice as 'Both'")
                        
                
        with tabs[3]:
            ragas_evaluation_tab(query, model_choice, single_model)           
        show_system_metrics(model_choice)

    
           
    

def handle_single_model_summary(model_choice, documents):
    """Handle summary generation for a single model."""
    
    handler = OpenAIHandler(OPENAI_API_KEY) if model_choice == "OpenAI" else NVIDIAHandler(NVIDIA_API_KEY)
    
    if st.button("Generate Summary", key="single_summary"):
        with st.spinner("Generating summary..."):
            if model_choice == "OpenAI":
               openai_summary, openai_summary_time = process_document(handler, " ".join(documents))
               st.session_state.openai_summary = openai_summary 
               st.session_state.openai_summary_time = openai_summary_time
            elif model_choice=="NVIDIA":
                nvidia_summary,nvidia_summary_time=process_document(handler," ".join(documents))
                st.session_state.nvidia_summary=nvidia_summary
                st.session_state.nvidia_summary_time=nvidia_summary_time
            
            
            # Store the summary and response time correctly
            # model_key = model_choice.lower()
            # st.session_state[f"{model_key}_summary"] = summary
            # st.session_state[f"{model_key}_summary_time"] = summary_time  # Fixed response time storage
    
        if f"{model_choice.lower()}_summary" in st.session_state:
            with st.container():
                st.markdown(f"""
                <div class="response-box">
                    <h4>{model_choice} Summary:</h4>
                    <p>{st.session_state[f"{model_choice.lower()}_summary"]}</p>
                    <small>Response Time: {st.session_state[f"{model_choice.lower()}_summary_time"]:.2f}s</small>
                </div>
                """, unsafe_allow_html=True)


def handle_both_models_summary(documents):
    """Handle summary generation for both models"""
    if st.button("Generate Summaries", key="both_summaries"):
        st.session_state.summary_processing=True
        st.session_state.comparison_result_summary=None
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("Generating OpenAI summary..."):
                openai_handler = OpenAIHandler(OPENAI_API_KEY)
                summary, openai_summary_time = process_document(openai_handler, " ".join(documents))
                st.session_state.openai_summary = summary
                st.session_state.openai_summary_time = openai_summary_time

                

        
        with col2:
            with st.spinner("Generating NVIDIA summary..."):
                nvidia_handler = NVIDIAHandler(NVIDIA_API_KEY)
                summary, nvidia_summary_time = process_document(nvidia_handler, " ".join(documents))
                st.session_state.nvidia_summary = summary
                st.session_state.nvidia_summary_time = nvidia_summary_time
    
    if st.session_state.openai_summary or st.session_state.nvidia_summary:
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.openai_summary:
                st.markdown("""
                <div class="response-box">
                    <h4>OpenAI Summary</h4>
                    <p>{}</p>
                </div>
                """.format(st.session_state.openai_summary), unsafe_allow_html=True)
        
        with col2:
            if st.session_state.nvidia_summary:
                st.markdown("""
                <div class="response-box">
                    <h4>NVIDIA Summary</h4>
                    <p>{}</p>
                </div>
                """.format(st.session_state.nvidia_summary), unsafe_allow_html=True)



def handle_query_processing(query, model_choice, documents, single_model, both_models):
    """Handle query processing and display."""

    if "query_processing" not in st.session_state:
        st.session_state.query_processing = False
    if "current_query" not in st.session_state:
        st.session_state.current_query = None

    if st.button("Submit Query", key="query_button"):
        if query:
            st.session_state.current_query = query
            st.session_state.query_processing = True
            st.session_state.comparison_result_response = None  # Clear previous comparison

            # ✅ Clear previous responses to avoid duplication
            if single_model:
                if model_choice == "OpenAI":
                    st.session_state.nvidia_response = None
                else:
                    st.session_state.openai_response = None

            with st.spinner("Processing query..."):
                if single_model:
                    handle_single_model_query(model_choice, documents, query)
                else:
                    handle_both_models_query(documents, query)

        else:
            st.warning("Please enter a query before submitting.")

    # ✅ Display only once (moved from handle_single_model_query and handle_both_models_query)
    if st.session_state.query_processing and st.session_state.current_query:
        if single_model:
            response_key = f"{model_choice.lower()}_response"
            response_time_key = f"{model_choice.lower()}_query_time"

            if st.session_state.get(response_key):
                st.markdown(f"""
                <div class="response-box">
                    <h4>{model_choice} Response</h4>
                    <p>{st.session_state[response_key]}</p>
                    <small>Response Time: {st.session_state.get(response_time_key, 0):.2f}s</small>
                </div>
                """, unsafe_allow_html=True)

        elif both_models:
            col1, col2 = st.columns(2)

            with col1:
                if st.session_state.get("openai_response"):
                    st.markdown(f"""
                    <div class="response-box">
                        <h4>OpenAI Response</h4>
                        <p>{st.session_state.openai_response}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                if st.session_state.get("nvidia_response"):
                    st.markdown(f"""
                    <div class="response-box">
                        <h4>NVIDIA Response</h4>
                        <p>{st.session_state.nvidia_response}</p>
                    </div>
                    """, unsafe_allow_html=True)


def handle_comparisons(documents, query):
    """Handle model output comparisons"""
    model_evaluation = ModelEvaluation(GEMINI_API_KEY)
    
    if (st.session_state.openai_summary is None and 
    st.session_state.nvidia_summary is None and 
    st.session_state.openai_response is None and 
    st.session_state.nvidia_response is None):
      st.warning("First generate summaries and responses.")

    
    if st.session_state.openai_summary and st.session_state.nvidia_summary:
        st.markdown("### Summary Comparison")
        if st.button("Compare Summaries", key="compare_summaries"):
            with st.spinner("Analyzing summaries..."):
                st.session_state.comparison_result_summary = model_evaluation.get_gemini_evaluation_summary(
                    st.session_state.openai_summary,
                    st.session_state.nvidia_summary,
                    documents
                )
    
    if st.session_state.comparison_result_summary:
        st.markdown(f"""
        <div class="response-box">
            <h4>Summary Comparison Analysis</h4>
            <p>{st.session_state.comparison_result_summary}</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    if st.session_state.openai_response and st.session_state.nvidia_response:
        st.markdown("### Response Comparison")
        if st.button("Compare Responses", key="compare_responses"):
            with st.spinner("Analyzing responses..."):
                st.session_state.comparison_result_response = model_evaluation.get_gemini_evaluation_qna(
                    st.session_state.openai_response,
                    st.session_state.nvidia_response,
                    documents,
                    query
                )
    
    if st.session_state.comparison_result_response:
        st.markdown(f"""
        <div class="response-box">
            <h4>Response Comparison Analysis</h4>
            <p>{st.session_state.comparison_result_response}</p>
        </div>
        """, unsafe_allow_html=True)
        

  
            

if __name__ == "__main__":
    main()
