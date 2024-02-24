'''
1) Get pdf , category and job description 
2) Filter pdf based on category 
3) Find cosine similarity for pdf and job description and rerank based on the output  


when submit button is clicked , we need to clean the pdf , and then pass the cleaned pdf to the category filtration in the form of dictionary 


upload pdf 


''' 

from torch import cuda 
from BERTClassifier import BERTClass , predict_text_class 
from JobDescriptionMatcher import PDFJobMatcher, cleanText
import streamlit as st 
import PyPDF2
import torch



def pdf_cleaning(pdf_list):
    for key,value in pdf_list.items():
        pdf_list[key] = cleanText(value) 
    
    return pdf_list


def getCategoryClass(category): 
    category_mapping_flipped_lowercase = {
        'accountant': 0,
        'advocate': 1,
        'agriculture': 2,
        'apparel': 3,
        'arts': 4,
        'automobile': 5,
        'aviation': 6,
        'banking': 7,
        'bpo': 8,
        'business-development': 9,
        'chef': 10,
        'construction': 11,
        'consultant': 12,
        'designer': 13,
        'digital-media': 14,
        'engineering': 15,
        'finance': 16,
        'fitness': 17,
        'healthcare': 18,
        'hr': 19,
        'information-technology': 20,
        'public-relations': 21,
        'sales': 22,
        'teacher': 23
    }

    return category_mapping_flipped_lowercase[category.lower()]   



def category_filtration(pdf_list,category_class):
    loaded_model = load_model(BERTClass, 'bert_model_trained.pth') 
    filtered_pdf = {} 

    for key,value in pdf_list.items():
        predicted_class = predict_text_class(value, loaded_model)
        if predicted_class == category_class:
            filtered_pdf[key] = value

    return filtered_pdf



def load_model(model_class, filepath):
    device='cuda' if cuda.is_available() else 'cpu' 
    model = model_class()
    model.load_state_dict(torch.load(filepath))
    model.to(device)
    model.eval() 
    return model


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text



def main():

    st.title('Resume Radar : Find Potential Candidate at One Click')

    #selections of category options 
    option = st.selectbox(
    "Category : ",
    ('Accountant', 'Advocate', 'Agriculture', 'Apparel', 'Arts',
    'Automobile', 'Aviation', 'Banking', 'BPO', 'Business-development', 'Chef', 'Construction',
    'Consultant', 'Designer', 'Digital-media', 'Engineering', 'Finance', 'Fitness', 'Healthcare', 
    'HR', 'Information-technology', 'Public-relations', 'Sales', 'Teacher'), 
    index=None,
    placeholder="Select a category",)

    #job description text area
    job_desc = st.text_area('Enter Job Description') 

    #similarity between job description and resume
    expected_similarity = st.slider('Minimum expected similarity', 0, 100, 0) 

    #stores the file name as key and the text as value
    pdf_text_pair = {}

 
    #upload pdf resumes
    uploaded_files = st.file_uploader("Choose resume file(s) : ", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        pdf_text_pair[uploaded_file.name] = extract_text_from_pdf(uploaded_file)  
 

    #submit button
    if st.button("Submit", type="secondary"): 
        if not (option and job_desc and uploaded_files):    
            st.warning("Please fill in all the fields and upload resume before clicking the 'Submit' button", icon="⚠️") 
        else: 
            #resume cleaning
            cleaned_pdf = pdf_cleaning(pdf_text_pair)

            #filtering resumes based on category 
            filtered_pdf = category_filtration(cleaned_pdf,getCategoryClass(option))  
            st.write("Possible matches : ",filtered_pdf.keys()) 
            
            #find matching jobs
            job_match = PDFJobMatcher(filtered_pdf,job_desc,expected_similarity).match() 
            st.write("Top match : ",job_match)  


if __name__ == '__main__': 
    main()  