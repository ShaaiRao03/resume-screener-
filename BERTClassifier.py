import transformers
import torch
from torch import cuda  

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 24)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output 
    

def predict_text_class(input_text, model):
    device='cuda' if cuda.is_available() else 'cpu' 
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
    return predicted_label  