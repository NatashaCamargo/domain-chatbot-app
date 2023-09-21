from langchain.prompts.prompt import PromptTemplate

_template = """Dado o histórico de conversa a seguir e uma pergunta adicional, reescreva a pergunta adicional para que esta seja uma pergunta independente
Caso você não saiba a resposta, não tente inventar uma, e não responda com alguma das respostas anteriores, apenas diga educadamente
que não foi possível encontrar a resposta.

histórico de conversa 
{chat_history}
Pergunta adicional: {question}
Pergunta independente:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)