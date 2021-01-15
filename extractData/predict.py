import torch

import argparse
from my_data import MyDataset, VOCAB
from my_models import MyModel0
from my_utils import pred_to_dict
import json

parser = argparse.ArgumentParser()
args = parser.parse_args()
model = MyModel0(len(VOCAB), 16, args.hidden_size).to(args.device)

model.load_state_dict(torch.load("model.pth"))
model.eval()
dataset = MyDataset(None, args.device, test_path="data/test_dict.pth")


def get_tensor(self,text):
    text_tensor = torch.zeros(len(text), 1, dtype=torch.long)
    text_tensor[:, 0] = torch.LongTensor([VOCAB.find(c) for c in text])

    return text_tensor.to(self.device)

def predict(path_text_data):
    text_data=open(path_text_data).read()

    with torch.no_grad():
        text_tensor = dataset.get_tensor(text_data)

        oupt=model(text_tensor)
        prob = torch.nn.functional.softmax(oupt, dim=2)
        prob, pred = torch.max(prob, dim=2)

        prob = prob.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()

        result=pred_to_dict(text_data,pred,prob)

    return result

if __name__ == "__main__":
    predict(args[0])






