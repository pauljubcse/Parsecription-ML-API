from time import time
import fitz  # PyMuPDF, imported as fitz for backward compatibility reasons
from PIL import Image
import pytesseract
import requests
from fastapi import FastAPI
from pydantic import BaseModel
# import mysql.connector
# import pyfiglet
# import os
# import openai
import json
# from langchain.agents import tool
from pydantic import BaseModel, Field
# from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
# from langchain.schema.output_parser import StrOutputParser
# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain.prompts import HumanMessagePromptTemplate
# from langchain_core.messages import SystemMessage
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.chat_models import ChatOpenAI
from typing import Optional
import re
import torch
from transformers import BertTokenizer, BertForTokenClassification
from utils import *

'''

file_path = "Report.pdf"
doc = fitz.open(file_path)  # open document
i=0
for page in doc:
    pix = page.get_pixmap()  # render page to an image
    pix.save(f"page_{i}.png")
    i+=1
'''
start_time_str = "07:00"
end_time_str = "21:00"
mid_time_str = "13:00"

# pdf_path = 'P1.pdf'


# # LLM CODE

# model = ChatOpenAI()


# functions = [
#     convert_pydantic_to_openai_function(MedInfo),
#     convert_pydantic_to_openai_function(PatientInfo),
#     # convert_pydantic_to_openai_function(DoctorInfo),
#     convert_pydantic_to_openai_function(OtherInfo)
# ]
# parser = model.bind(functions=functions)

# # BERT FOR NER
# tokenizer = BertTokenizer.from_pretrained(
#     'medical-ner-proj/bert-medical-ner-proj')
# model = BertForTokenClassification.from_pretrained(
#     'medical-ner-proj/bert-medical-ner-proj')
# # NER Labels
# id2label = {
#     "0": "B_person",
#     "1": "B_problem",
#     "2": "B_pronoun",
#     "3": "B_test",
#     "4": "B_treatment",
#     "5": "I_person",
#     "6": "I_problem",
#     "7": "I_test",
#     "8": "I_treatment",
#     "9": "O"
# }


# def filterLines(textLines):
#     result = []
#     for textLine in textLines:
#         tokenized_input = tokenizer(textLine, return_tensors="pt")
#         outputs = model(**tokenized_input)
#         # Process the outputs to get NER predictions
#         predictions = torch.argmax(outputs.logits, dim=2)
#         for i in range(len(predictions[0])):
#             # print("Token: ", textLine[i], " Label: ",id2label[str(predictions[0][i].item())])
#             if id2label[str(predictions[0][i].item())] == 'B_test' or id2label[str(predictions[0][i].item())] == 'B_treatment':
#                 result.append(textLine)
#                 # print(textLine)
#                 break
#     return result


class Request(BaseModel):
    url: str
    isPrescription: bool


parser = ParserModel()
nerFilter = NERFilter()
explainerModel = ExplainerModel()
app = FastAPI()


@app.get("/")
def welcome():
    return "Welcome to Parsecription API"


@app.get("/test")
def test():
    return {"Status": "Running"}


@app.post("/parse-prescription")
def parsePrescription(R: Request):
    start = time()
    url = R.url
    destination_path = 'P1.pdf'
    download_pdf(url, destination_path)
    text = pdf_to_text(destination_path)

    # Define the replacement patterns
    patterns = {'tab': 'tablet', 'cap': 'capsule',
                "TAB": "tablet", "Cap": "capsule", "Tab": "tablet"}
    # Create a regular expression pattern for matching the keys in the dictionary
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key)
                         for key in patterns.keys()) + r')\b')
    # Replace the matched patterns with their corresponding values
    text = pattern.sub(lambda match: patterns[match.group(0)], text)
    textSplit = splitter(text)

    filteredTextLines = nerFilter.filter(textSplit)
    ParsedResult = {'MedInfo': [], 'PatientInfo': []}

    for i in filteredTextLines:
        response = parser.invoke(i)
        # print(response)
        if (response.content == ''):
            outer_dict = response.additional_kwargs['function_call']
            arguments_dict = json.loads(outer_dict['arguments'])
            outer_dict['arguments'] = arguments_dict
            try:
                if (outer_dict['name'] == 'MedInfo' and arguments_dict['frequency']):
                    # print(outer_dict)
                    ParsedResult[outer_dict['name']].append(
                        outer_dict['arguments'])
                elif (outer_dict['name'] == 'PatientInfo'):
                    # print(outer_dict)
                    ParsedResult[outer_dict['name']].append(
                        outer_dict['arguments'])

            except:
                if (outer_dict['name'] == 'PatientInfo'):
                    # print(outer_dict)
                    ParsedResult[outer_dict['name']].append(
                        outer_dict['arguments'])

    # return json.dumps(ParsedResult)

    for i in range(len(ParsedResult['MedInfo'])):
        # print("Entry")
        # print(ParsedResult['MedInfo'][i])
        freq = ParsedResult['MedInfo'][i]["frequency"]
        # print(freq)
        L = generate_timestamps(
            start_time_str=start_time_str, end_time_str=end_time_str, mid_time_str=mid_time_str, n=freq)
        # print("Entry2")
        print(L)
        ParsedResult['MedInfo'][i]['timestamps'] = L

    try:
        weight = {}
        height = {}
        bp = {}
        bloodSugar = {}
        pulseRate = {}
        for i in ParsedResult['PatientInfo']:
            try:
                height['val0'] = i['height']
            except:
                pass
            try:
                weight['val0'] = i['weight']
            except:
                pass
            try:
                bp['val0'] = i['bp_systolic']
                bp['val1'] = i['bp_diastolic']
            except:
                pass
            try:
                bloodSugar['val0'] = i['blood_sugar_pp']
                bloodSugar['val1'] = i['blood_sugar_fasting']
            except:
                pass
            try:
                pulseRate['val0'] = i['heart_rate']
            except:
                pass

        ParsedResult['weight'] = weight
        ParsedResult['height'] = height
        ParsedResult['pulseRate'] = pulseRate
        ParsedResult['bp'] = bp
        ParsedResult['bloodSugar'] = bloodSugar
    except:
        pass

    end = time()
    ParsedResult['processingTime'] = end-start
    return ParsedResult


@app.post("/parse-report")
def parseReport(R: Request):
    start = time()
    url = R.url
    destination_path = 'Report.pdf'
    download_pdf(url, destination_path)
    text = pdf_to_text(destination_path)
    result = explainerModel.invoke(text)
    end = time()
    return {
        "summary": result,
        "processingTime": end-start
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.208.86", port=8080)
