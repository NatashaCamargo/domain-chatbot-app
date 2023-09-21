from langchain.prompts import PromptTemplate

template = """{summaries}
Por favor, responda a questão utilizando apenas a informação presentes no documento fornecido, avalie 
também se a pergunta tem alguma relação com o a Pergunta feita abaixo, caso não exista
respondaa educadamente que não foi possível encontrar a resposta.
Inclua informações sobre as fontes que você utilizou para criar a resposta se elas forem relevantes
como resposta, se a resposta não parecer coerente ou nada for encontrado responda
educadamente explicando que não foi possível encontrar a resposta.

Pergunta:{question}
Answer:
"""

PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
