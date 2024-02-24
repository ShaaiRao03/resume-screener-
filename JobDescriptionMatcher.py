from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

import re  
import nltk
nltk.download('stopwords') 
stop = stopwords.words('english') 


class PDFJobMatcher: 
    def __init__(self, pdfs, job_description, similarity):
        self.pdfs = pdfs
        self.job_description = job_description
        self.similarity = similarity

    def match(self):
        file_cos_pair = generate_cosine_similarity(self.pdfs,self.job_description)
        sorted_file_val_pair = dict(sorted(file_cos_pair.items(), key=lambda item: item[1], reverse=True))
        
        matched_pdfs = {} 
        for pdf_name,val in sorted_file_val_pair.items():
            if val >= self.similarity : 
                matched_pdfs[pdf_name] = val  
                    
        return matched_pdfs 



def generate_cosine_similarity(pdf_list,job_description):   

    preprocessed_pdf_list = preprocess_pdf_text(pdf_list)  

    vectorizer = TfidfVectorizer()
    vectorizer.fit([cleanText(job_description)])   

    cosine_similarity_pair = {} 

    X = vectorizer.transform([job_description]) 

    for pdf_name, text in preprocessed_pdf_list.items(): 
        Y = vectorizer.transform([text])
        cosine_similarity_pair[pdf_name] = round(cosine_similarity(X,Y)[0][0]*100,2) 

    return cosine_similarity_pair



def cleanText(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]','',text) 
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'\n', ' ' ,text) 
    clean_text_lambda = lambda x: ' '.join([word for word in x.split() if word not in stop])
    return clean_text_lambda(text) 
 

def preprocess_pdf_text(pdf_list):
    for pdf_name , text in pdf_list.items(): 
        pdf_list[pdf_name] = cleanText(text)
    return pdf_list 
