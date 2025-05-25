from langchain_openai import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
#from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains.summarize import load_summarize_chain
#from langchain.chains import RetrievalQA
import os
import json
import re
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"
#MODEL_NAME = 'gpt-3.5-turbo'

def clean_label(text):
    # Remove labels like "A.", "A)", "B.", "B)", etc.
    return re.sub(r'^[A-E][.)]\s*', '', text.strip())

def convert_content_to_questions(content, ques_gen_chain):
    ques_list = []
    #print("Content = ", content)
    doc = [Document(page_content=content)]
    ques = ques_gen_chain.invoke(doc)
    print("Questions = ")
    print(ques['output_text'])

    # Clean the output_text by removing ```json and ```
    cleaned_ques = ques['output_text'].replace('```json', '').replace('```', '').strip()

    # Handle common escape sequences
    cleaned_ques = cleaned_ques.replace('(\\)', '(\\\\)')
    #print("cleaned_ques =", cleaned_ques)

    # Strip leading and trailing whitespace
    cleaned_ques = cleaned_ques.strip()

    # Split the cleaned JSON string into individual questions
    # Assuming the questions are separated by '},{' and wrapped in an array
    if cleaned_ques.startswith('[') and cleaned_ques.endswith(']'):
        cleaned_ques = cleaned_ques[1:-1]  # Remove the surrounding square brackets
    
    individual_questions = re.split(r'},\s+{', cleaned_ques)
    #for iq in individual_questions:
    #    print("individual_question before = ", iq) 
    
    # Add the curly braces back to each individual question
    individual_questions = [q if q.strip().startswith('{') else '{' + q for q in individual_questions]
    individual_questions = [q if q.strip().endswith('}') else q + '}' for q in individual_questions]
    #individual_questions = [q if q.startswith('{') else '{' + q for q in individual_questions]
    #individual_questions = [q if q.endswith('}') else q + '}' for q in individual_questions]

    #for iq in individual_questions:
    #    print("individual_question after = ", iq) 

    for question in individual_questions:
        try:
            question_dict = json.loads(question)
            # Scramble the order of the options
            if 'options' in question_dict:
                print("Before shuffle: ", question_dict['options'])
                question_dict['options'] = [clean_label(option) for option in question_dict['options']]
                random.shuffle(question_dict['options'])
                print("After shuffle: ", question_dict['options'])
            if 'answer' in question_dict:
                question_dict['answer'] = clean_label(question_dict['answer'])
            ques_list.append(question_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for question: {e}")
            print(f"Skipping the question: {question}")
    
    print("ques_list length = ", len(ques_list))
    return ques_list

def generate_questions(file_path, pagenum):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    #print(data)

    question_gen = ''

    #for page in data:
    #    print("Page = ", page)
    #    print("")
        #question_gen += page.page_content

    #document_ques_gen = [Document(page_content=t.page_content, metadata=t.metadata) for t in data]

    #for doc in document_ques_gen:
    #    print("Doc = ", doc)
    #    print("")

    llm_ques_gen_pipeline = ChatOpenAI(
        temperature = 0.3,
        model = MODEL_NAME,
    )
    #    model = "gpt-3.5-turbo"

    # Set the type of question to generate here
    # Can be left blank (for open ended question) or "true/false", "multiple choice"
    question_type = "multiple choice"

    prompt_template = """
You are an expert at creating questions based on class materials and teacher handouts.
Your goal is to prepare a student for their exam.
You do this by asking """ + question_type + """ questions about the text below:

------------
{text}
------------

Create """ + question_type + """ questions that will prepare the student for their tests.
Make sure not to lose any important information.  Do not refer to the given text in the 
question, because the student will not have access to it.
Please provide the questions in json list format where each entries in the list has a 'text' entry
which contains the question text, an 'options' entry which contains the options for multiple choice,
and an 'answer' entry which contains the correct answer. Do not label the options or answers with letters.

QUESTIONS:
"""

    prompt_template2 = """
You are an expert at creating AP style multiple choice questions based on class materials and teacher handouts.
Your goal is to prepare a student for their AP Computer Science exam.
You do this by asking """ + question_type + """ questions about the text below:

------------
{text}
------------

Create sample AP computer science multiple choice questions related to the topics in the above text.
They should be of similar difficulty as the AP test. Do not refer to the given text in the 
question, because the student will not have access to it.
Please provide the questions in json list format where each entries in the list has a 'text' entry
which contains the question text, an 'options' entry which contains the options for multiple choice,
and an 'answer' entry which contains the correct answer. Do not label the options or answers with letters.

QUESTIONS:
"""

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template2, input_variables=["text"])

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                              chain_type = "refine", 
                                              #verbose = True, 
                                              question_prompt=PROMPT_QUESTIONS)
                                              #refine_prompt=REFINE_PROMPT)

    splitter_ques_gen = TokenTextSplitter(
        model_name = MODEL_NAME,
        chunk_size = 300,
        chunk_overlap = 50
    )

    all_questions = []
    accum_page_content = ""

    # If you only want to generate questions for a subset of the doc, put the page range here
    # data[:] - processes all the pages
    # data[0:4] - processes the first 5 pages
    # data[5:] - processes page 6 to the end
    data_subset = data[:]

    for page in data_subset:
        print("Page = ", page)
        print("")
        print("Pagenum = page.metadata['page'] = ", page.metadata['page'])
        if (-1 not in pagenum and page.metadata['page'] not in pagenum):
            continue # Skip this page

        print("Length of page.page_content =", len(page.page_content))
        accum_page_content += page.page_content
        if (len(accum_page_content) > 1000):
            # Split into chunks
            chunks_ques_gen = splitter_ques_gen.split_text(accum_page_content)

            for chunk in chunks_ques_gen:
                print("Chunk = ", chunk)
                print("")

                ques_list = convert_content_to_questions(chunk, ques_gen_chain)
                all_questions.extend(ques_list)

                print("---------------")
                accum_page_content = ""
                print("")

        if (len(accum_page_content) > 100):
            ques_list = convert_content_to_questions(accum_page_content, ques_gen_chain)
            all_questions.extend(ques_list)

            print("---------------")
            accum_page_content = ""
            print("")

    print("Length of all_questions = ", len(all_questions))
    return all_questions
