import pandas as pd
from db_call import QASystem
import pandas as pd
import csv

def chatbot(input):
    # First try to get a response from the QASystem
    # Initialize the QASystem
    qa = QASystem('test.csv')
    try:
        qa_response = qa.get_response(input)
        print("Try")
    except:
        qa_response = "I can't answer this question."
        print("Except")
        # Save the question and the AI's response to the CSV file
        data = {'Questions': [input], 'Answers': [qa_response]}
        df = pd.DataFrame(data)
        #print(data)
        print("File path:", r'C:\\Users\nikc\\websitechatbot1\\questionsCollection.csv')
        df.to_csv(r'C:\Users\nikc\OneDrive - Prairie View A&M University\Desktop\Langchain\questionsCollection.csv', mode='a', header=False, index=False)
       
    return qa_response