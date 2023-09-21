import os
import pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains.llm import LLMChain
from chat_history_prompt import CONDENSE_QUESTION_PROMPT
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate

from customprompt import PROMPT

# import sys
# #sys.path.append('C:/Users/noteb/Documents/MVP/mvp/utils')
# sys.path.insert(1, '/mvp/utils')
# os.environ['PYTHONSAFEPATH'] = 'C:/Users/noteb/Documents/MVP/mvp/utils'

class LLMHelper:
    def __init__(self,
    embeddings: OpenAIEmbeddings = None,
    llm: ChatOpenAI = None,
    llm_type: str = None,
    temperature: float = None,
    custom_prompt: str = "",
    vector_store: Pinecone = None,
    vector_store_type: str = None):

        # Set API keys
        self.pinecone_api_key = os.environ["PINECONE_API_KEY"]
        self.pinecone_api_env = os.environ["PINECONE_API_ENV"]
        self.openai_api_key = os.environ["OPENAI_API_KEY"]

        # OpenAI settings
        self.embedding_model: str = os.getenv("OPENAI_EMBEDDINGS_ENGINE", "text-embedding-ada-002")
        self.temperature: float = float(os.getenv("OPENAI_TEMPERATURE", 0.0)) if temperature is None else temperature
        self.vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "Pinecone") if vector_store_type is None else vector_store_type
        self.llm_type: str = os.getenv("LLM_TYPE","Chat" ) if llm_type is None else llm_type
        self.prompt = PROMPT if custom_prompt == '' else PromptTemplate(template=custom_prompt, input_variables=["summaries", "question"])
        self.embedding_model = "text-embedding-ada-002"

        # setting embedding model
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=self.embedding_model, openai_api_key=self.openai_api_key) if embeddings is None else embeddings
        # setting the llm model
        if self.llm_type == "Chat":
            self.llm: ChatOpenAI = ChatOpenAI(temperature=self.temperature, openai_api_key=self.openai_api_key) if llm is None else llm
        else:
            raise Exception("Currently only the ChatGPT 3.5 turbo model is allowed.")
        
        # vector store settings
        self.vector_store = vector_store

    def set_vector_store(self, index_name=None):
        """
            This is a setter function that aims to set our vector store.
            It initializes the Vector Store and them creates a vector store instance.
            Atributes:
                - index_name = reffers to the vector database's name
        """
        # setting the vector database
        if self.vector_store_type == "Pinecone":
            pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_api_env)
            self.vector_store: Pinecone = Pinecone.from_existing_index(index_name, self.embeddings)
        else:
            raise Exception("Currently we only support Pinecone vector database.")

        return self.vector_store
    
    def get_semantic_search_conversational_chain(self, question, chat_history):
        self.set_vector_store("source-test")
        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=False)
        # vector_store = self.set_vector_store("index name goes heree") # set index name to test
        doc_chain = load_qa_with_sources_chain(self.llm, chain_type="stuff", verbose=True, prompt=self.prompt)

        
        chain = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,     
            return_source_documents=False
        )

        result = chain({"question": question, "chat_history": chat_history})


        return question, result['answer']
