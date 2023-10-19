import sys
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
import os

"""
!pip install langchain 
!pip install faiss-cpu 
!pip install openai
!pip install unstructured
!pip install tiktoken
"""


# Import Langchain libraries
import pickle
import faiss
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

def preprocess_with_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

class QASystem:
    def is_greeting(self, text):
        greetings = ["hello", "hi", "hey", "howdy", "hola", "good morning", "good afternoon", "good evening", "g'day", "greetings", "salutations", "what's up", "how's it going", "how are you", "yo", "hi there", "hey there"]
        tokens = nltk.word_tokenize(text.lower())
        for token in tokens:
            if token in greetings:
                return True
        return False

    def is_goodbye(self, text):
        goodbyes = ["bye", "goodbye", "farewell", "see you", "see ya", "take care", "have a nice day", "so long", "catch you later", "until next time", "adieu", "later", "bye-bye", "cheerio", "ciao", "ta-ta", "see you later", "goodnight"]
        tokens = nltk.word_tokenize(text.lower())
        for token in tokens:
            if token in goodbyes:
                return True
        return False

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, escapechar='\\')
        self.questions_list = self.df['Questions'].tolist()
        self.answers_list = self.df['Answers'].tolist()
        self.vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
        self.X = self.vectorizer.fit_transform([preprocess_with_stopwords(q) for q in self.questions_list])
        
        

    def get_response(self, text):
        if self.is_greeting(text):
            return "Hello! How can I assist you today?"

        if self.is_goodbye(text):
            return "Goodbye! Have a great day."
        processed_text = preprocess_with_stopwords(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        similarities = cosine_similarity(vectorized_text, self.X)
        max_similarity = np.max(similarities)

        if max_similarity > 0.8:
            high_similarity_questions = [q for q, s in zip(self.questions_list, similarities[0]) if s > 0.8]
            target_answers = []

            for q in high_similarity_questions:
                q_index = self.questions_list.index(q)
                target_answers.append(self.answers_list[q_index])

            Z = self.vectorizer.fit_transform([preprocess_with_stopwords(q) for q in high_similarity_questions])
            processed_text_with_stopwords = preprocess_with_stopwords(text)
            vectorized_text_with_stopwords = self.vectorizer.transform([processed_text_with_stopwords])
            final_similarities = cosine_similarity(vectorized_text_with_stopwords, Z)
            closest = np.argmax(final_similarities)
            confidence = final_similarities[0, closest]

            if confidence >= 0.8:
                return target_answers[closest]

        # If confidence is below the threshold or if Langchain chatbot is needed
        # Call the Langchain chatbot and return its response
        langchain_response = self.call_langchain_chatbot(text)
        return langchain_response

    def call_langchain_chatbot(self, text):
        # Set your OpenAI API key
        os.environ["OPENAI_API_KEY"] = "*******"
        
        # List of URLs to load text data from
        urls = [
            'https://aitc.pvamu.edu/index.html',
            'https://aitc.pvamu.edu/index.html#aboutSection',
            'https://aitc.pvamu.edu/index.html#services',
            'https://aitc.pvamu.edu/index.html#internalApplication',
            'https://aitc.pvamu.edu/index.html#contactSection'
        ]

        # Load text data from the specified URLs
        loaders = UnstructuredURLLoader(urls=urls)
        data = loaders.load()

        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(
            #separator='\n',
            #chunk_size=1000,
            #chunk_overlap=200
        )

        # Split the documents into chunks
        docs = text_splitter.split_documents(data)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings()

        # Create a vector store using FAISS
        vectorStore_openAI = FAISS.from_documents(docs, embeddings) #WILL STORE VECTOR REPRESENTATION OF THE DOCUMENTS

        # Save the vector store to a file
        #with open("faiss_store_openai.pkl", "wb") as f: #This line opens a file named "faiss_store_openai.pkl" in binary write mode, and it is intended to save the vectorStore_openAI to this file.
                   #pickle.dump(vectorStore_openAI, f) # pickle module to serialize and save the vectorStore_openAI to the file. 
                   # This allows you to store the vector store and its associated vectors for later use without having to recompute the embeddings or vectors each time you run the code
            
                
        with open("faiss_store_openai.pkl", "rb") as f:
            vectorStore_openAI = pickle.load(f)    
        # Check if vectorStore_openAI is not None
        if vectorStore_openAI is not None:
            print("Vector Store loaded successfully.")
            # You can also inspect the contents if you want to:
            print(vectorStore_openAI)
        else:
            print("Vector Store failed to load. Check your file path or the file's integrity.")
        # Initialize Langchain's OpenAI model
        llm = OpenAI(temperature=0) # for generating natural language

        # Create a chatbot chain with the retriever
        #This chain is an integral part of the Langchain framework, and it's responsible for retrieving relevant information from your vector store (FAISS) based on the user's questions
        #llm is your Langchain language model (e.g., the OpenAI model).
        #vectorStore_openAI is your vector store created using FAISS. It contains vector representations of the documents that you want to retrieve information from.
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore_openAI.as_retriever())

        # Query the chatbot with the provided text
        response = chain({"question": text}, return_only_outputs=True)

        # Extract the answer from the response
        langchain_response = response["answer"]

        return langchain_response

   
