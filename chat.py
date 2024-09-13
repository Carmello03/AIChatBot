import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE, weights_only=False)  # Add weights_only argument

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state_dict = data['model_state_dict']

# Initialize model with the correct sizes
model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)
model.load_state_dict(model_state_dict)
model.eval()

bot_name = "Milana"
print(f"Hi i am {bot_name}, Let's chat! Type 'quit' to exit")

while True:
    sentence = input("You > ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Sorry, I don't understand.")
