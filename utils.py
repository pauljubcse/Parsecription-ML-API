# Dependancies
from datetime import datetime, timedelta
import fitz  # PyMuPDF, imported as fitz for backward compatibility reasons
from PIL import Image
import pytesseract
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import json
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
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
from langchain.schema.output_parser import StrOutputParser
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


def generate_timestamps(start_time_str, mid_time_str, end_time_str, n):
    # Convert start and end time strings to datetime objects
    # print("TIME")
    start_time = datetime.strptime(start_time_str, "%H:%M")
    end_time = datetime.strptime(end_time_str, "%H:%M")
    mid_time = datetime.strptime(mid_time_str, "%H:%M")

    # If n is 1, calculate the average of start and end time
    if n == 1:
        return [mid_time.strftime("%H:%M")]

    # If n is 2, return start and end timestamps
    elif n == 2:
        return [start_time.strftime("%H:%M"), end_time.strftime("%H:%M")]

    # For n > 2, calculate time interval between timestamps
    else:
        # Calculate time difference between start and end time
        time_diff = end_time - start_time

        # Calculate time interval between timestamps
        interval = time_diff / (n - 1)

        # Generate N timestamps
        timestamps = [start_time + i * interval for i in range(n)]

        # Format timestamps as strings in 24-hour format
        timestamps_str = [time.strftime("%H:%M") for time in timestamps]

        return timestamps_str


def splitter(text):
    result = []
    for i in text.split("\n"):
        if i not in [' ', '']:
            result.append(i)
    return result


def pixmap_to_image(pixmap):
    img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    img = img.convert("L")
    img.show()
    return img


def pdf_to_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)  # open document
    for page in doc:
        pix = page.get_pixmap()  # render page to an image
        image = pixmap_to_image(pix)
        text = text+pytesseract.image_to_string(image)
    return text


def download_pdf(url, destination):
    response = requests.get(url)

    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded successfully to {destination}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")


# Data Validation Classes
class PatientInfo(BaseModel):
    """Call this with a Line that contains Weight, BP, Heart Beat Rate BPM, O2 Saturation Levels etc"""
    weight: Optional[int] = Field(description="Weight in kg or pounds")
    height: Optional[int] = Field(description="Height in cm")
    bp_systolic: Optional[int] = Field(
        description="Systolic Blood Pressure or BP in mm/Hg")
    bp_diastolic: Optional[int] = Field(
        description="Diastolic Blood Pressure or BP in mm/Hg")
    heart_rate: Optional[int] = Field(description="Heart Bate Rate")
    blood_sugar_pp: Optional[int] = Field(description="Blood Sugar Level PP")
    blood_sugar_fasting: Optional[int] = Field(
        description="Blood Sugar Level Fasting")


class OtherInfo(BaseModel):
    """Diseases, Any Jibberish Text"""
    irrelevant: Optional[int] = Field(
        description="Diseases, random words, or noisy non-English words")


class DoctorInfo(BaseModel):
    """Call this with a Line that contains Doctors Name Explicitly Mentioned"""
    name: Optional[str] = Field(description="Name of Doctor, Do NOT Guess")


class MedInfo(BaseModel):
    """Call this with a Line that contains Medicine Name, Dosage, Frequency
    Avoid Generic Words like Medicine, Fever. Do not Guess."""
    name: Optional[str] = Field(description="Name of Medicine and Dosage(mg)")
    # dosage: Optional[str] = Field(description="Dosage of Medicine, can be empty")
    frequency: Optional[int] = Field(
        description="Frequency of Medicine Eg: Once a Day, Twice a Day Converted to Integer")
    number_of_days: Optional[int] = Field(
        description="Empty Unless Mentioned for How Many Days, Eg: for 7 days")
    specific_part_of_day: Optional[str] = Field(
        description="Do not Guess unless Mentioned. Specific Part of Day like Waking Up, Breakfast, Lunch, Dinner, Bedtime.")


class NERFilter():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            'medical-ner-proj/bert-medical-ner-proj')
        self.model = BertForTokenClassification.from_pretrained(
            'medical-ner-proj/bert-medical-ner-proj')
        self.id2label = {
            "0": "B_person",
            "1": "B_problem",
            "2": "B_pronoun",
            "3": "B_test",
            "4": "B_treatment",
            "5": "I_person",
            "6": "I_problem",
            "7": "I_test",
            "8": "I_treatment",
            "9": "O"
        }

    def filter(self, textLines):
        "Pass a List of Lines to be filtered"
        result = []
        for textLine in textLines:
            tokenized_input = self.tokenizer(textLine, return_tensors="pt")
            outputs = self.model(**tokenized_input)
            # Process the outputs to get NER predictions
            predictions = torch.argmax(outputs.logits, dim=2)
            for i in range(len(predictions[0])):
                # print("Token: ", textLine[i], " Label: ",id2label[str(predictions[0][i].item())])
                if self.id2label[str(predictions[0][i].item())] == 'B_test' or self.id2label[str(predictions[0][i].item())] == 'B_treatment':
                    result.append(textLine)
                    # print(textLine)
                    break

        return result


class ParserModel():
    def __init__(self):
        self.model = ChatOpenAI()
        self.functions = [
            convert_pydantic_to_openai_function(MedInfo),
            convert_pydantic_to_openai_function(PatientInfo),
            # convert_pydantic_to_openai_function(DoctorInfo),
            convert_pydantic_to_openai_function(OtherInfo)
        ]
        self.model = self.model.bind(functions=self.functions)

    def invoke(self, textLine):
        """Pass a Line of Text"""
        return self.model.invoke(textLine)


class ExplainerModel():
    def __init__(self):
        self.model = ChatOpenAI(temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Medical Assistant, Summarise the Findings of the following Pathological Report provided as text."),
            ("human", "Report: {report}")
        ])
        self.chain = self.prompt | self.model | StrOutputParser()

    def invoke(self, text):
        return self.chain.invoke({"report": text})
