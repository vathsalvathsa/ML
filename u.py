import os
os.environ["OPEN_API_KEY"] = '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'
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
llm_restro = OpenAI(temperature=0.6)
prompt_template_resto = PromptTemplate(
    input_variables=['age','gender','weight','height','veg_or_nonveg','disease','region','allengics','foodtype'],
    template="Diet Recommendation System:\n"
    "I want you to recommend 100 restaurents names, 10 breakfast names, 9 lunches names, 4 pre_workout names, 4post_workout names, 5 snacks names, 4 dinner names, 10 workout names"
    "based on the following criteria:\n"
    "Person age: {age}\n"
    "Person gender: {gender}\n"
    "Person height: {height}\n"
    "Person  veg_or_nonveg: {veg_or_nonveg}\n"
    "Person generic disease: {disease}\n"
    "Person region: {region}\n"
    "Person allergies : {allergies}\n"
    "Person foodtype: {foodtype}\n"
)
chain = LLMChain(llm=llm_restro,prompt=prompt_template_resto)