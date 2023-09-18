import random
import json
import torch
import nltk_utils
from nltk_utils import tokenization,bag_of_words
from model import NeuralNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) # learned parameters
model.eval()

bot_name = "Rico"
print("Hello ! I'm here to help you. Type 'quit if you want to exit.")


def get_response(msg):
    user_input = tokenization(msg)
    X = bag_of_words(user_input,all_words) # returns a numpy array of shape (no. of words in all_words)
    X = X.reshape(1,X.shape[0]) # 1 row and columns = words in all_words
    X = torch.from_numpy(X)  # convert int from matrix of shape (1, no. of words in all_words) to torch tensor as it is the right format for the model to be trained

    # getting out output
    output = model(X)

    _,predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()] # getting the actual tag by indexing it with the index returned from predictd.item()

    # check if the probability of this tag is high enough
    probs = torch.softmax(output, dim=1)  # lambda function to calculate probabiity of output using soft max function
    probability = probs[0][predicted.item()]  # the probability of the output

    if probability.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent["responses"])

    return "Sorry. I don't understand you..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(f"Nice to chat with you. {resp}")




