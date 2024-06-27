{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "2e506bed-c51f-4cc5-b75b-252367d31905",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import os\n",
    "# os.environ['REQUESTS_CA_BUNDLE'] = r'C:\\One-Drive\\OneDrive - Tredence\\Documents\\LLM Engineer Track Program\\Capstone Project\\Zscaler Root CA.crt'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "1922acfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_community.document_loaders import WikipediaLoader\n",
    "from langchain_community.vectorstores import FAISS\n",
    "from langchain_openai import OpenAIEmbeddings\n",
    "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
    "from langchain_community.document_loaders import WebBaseLoader\n",
    "from langchain_core.prompts import PromptTemplate\n",
    "from langchain.chains.question_answering import load_qa_chain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "f1c936d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "from datetime import datetime\n",
    "from dotted_dict import DottedDict\n",
    "from langchain_community.vectorstores import Chroma\n",
    "from langchain_openai import AzureChatOpenAI\n",
    "from langchain_openai import AzureOpenAIEmbeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "02dce943",
   "metadata": {},
   "outputs": [],
   "source": [
    "azure_config = {\n",
    "    \"model_deployment\": \"GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO\",\n",
    "    \"embedding_deployment\": \"ADA_RAG_DONO_DEMO\",\n",
    "    \"embedding_name\": \"text-embedding-ada-002\",\n",
    "    \"api_key\":\"f6f4b8aec16b4094bfe0b8e063dbf1a3\",\n",
    "    \"api_version\":\"2024-02-01\",\n",
    "    \"endpoint\":\"https://dono-rag-demo-resource-instance.openai.azure.com/\",\n",
    "    \"model\":\"GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO\"\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "e8d076ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "def models(azure_config):\n",
    "    print(\"developing models\")\n",
    "    models=DottedDict()\n",
    "    llm = AzureChatOpenAI(temperature=0,\n",
    "                      api_key=azure_config[\"api_key\"],\n",
    "                      openai_api_version=azure_config[\"api_version\"],\n",
    "                      azure_endpoint=azure_config[\"endpoint\"],\n",
    "                      model=azure_config[\"model_deployment\"],\n",
    "                      validate_base_url=False)\n",
    "    embedding_model = AzureOpenAIEmbeddings(\n",
    "        api_key=azure_config[\"api_key\"],\n",
    "        openai_api_version=azure_config[\"api_version\"],\n",
    "        azure_endpoint=azure_config[\"endpoint\"],\n",
    "        model = azure_config[\"embedding_deployment\"]\n",
    "    )\n",
    "    models.llm=llm\n",
    "    models.embedding_model=embedding_model \n",
    "    return models\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "4055d7cb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "developing models\n"
     ]
    }
   ],
   "source": [
    "models=models(azure_config)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "11108c3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_wikipedia(query):\n",
    "    from langchain_community.document_loaders import WikipediaLoader\n",
    "    loader = WebBaseLoader(web_path=query)\n",
    "    data = loader.load()\n",
    "    return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "685eb36e",
   "metadata": {},
   "outputs": [],
   "source": [
    "doc=load_wikipedia('https://en.wikipedia.org/wiki/FIFA_World_Cup')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "45cae8de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "list"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(doc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "635ae325",
   "metadata": {},
   "outputs": [],
   "source": [
    "def chunking(data):\n",
    "    from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
    "    text_splitter =RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30, length_function=len, is_separator_regex=False, separators=[\".\"])\n",
    "    chunk = text_splitter.split_documents(data)\n",
    "    return chunk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "6bab2a42",
   "metadata": {},
   "outputs": [],
   "source": [
    "chunked_data=chunking(doc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "8acf07be",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_vector_store(text_chunks, models):\n",
    "    embeddings = models.embedding_model\n",
    "    vectordb=FAISS.from_documents(chunked_data,embeddings)\n",
    "    return vectordb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "b41c917c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_conversational_chain(llm):\n",
    "    prompt_template = \"\"\"\n",
    "    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in\n",
    "    provided context just say, \"answer is not available in the context\", don't provide the wrong answer\\n\\n\n",
    "    Context:\\n {context}?\\n\n",
    "    Question: \\n{question}\\n\n",
    "\n",
    "    Answer:\n",
    "    \"\"\"\n",
    "    prompt = PromptTemplate(template=prompt_template, input_variables=[\"context\", \"question\"])\n",
    "    chain = load_qa_chain(llm, chain_type=\"stuff\", prompt=prompt)\n",
    "    return chain\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "b31996d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_answer(query):\n",
    "    document_search=get_vector_store(chunked, models)\n",
    "    similar_docs = document_search.similarity_search(query, k=1) # get closest chunks\n",
    "    chain = get_conversational_chain(models.llm)\n",
    "    answer = chain.invoke(input={\"input_documents\": similar_docs, \"question\": query}, return_only_outputs=True)\n",
    "    return answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "b4d7f06f",
   "metadata": {},
   "outputs": [],
   "source": [
    "answer = get_answer('Who was the host of 1982 fifa world cup')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "47f4f39f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'output_text': 'Spain was the host of the 1982 FIFA World Cup.'}"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "d9e19356",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.memory import ChatMessageHistory\n",
    "history = ChatMessageHistory()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3cfca95-0e06-4962-b1ec-4e19cb57d3f1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
