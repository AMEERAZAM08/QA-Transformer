import torch
import torch.nn as nn
import torch.optim as optim

# # Example data
# story = """Once upon a time in a faraway kingdom, there was a brave knight named Arthur.
#         Arthur protected the kingdom from evil forces and was loved by the people. 
#         He had a loyal friend, Merlin, who was a wise wizard."""

# qa_pairs = [
#     ("Who was the brave knight?", "Arthur"),
#     ("Who was Arthur's friend?", "Merlin"),
#     ("What did Arthur protect?", "kingdom"),
#     ("Who was loved by the people?", "Arthur")
# ]
# Long story provided
story = """
In a distant realm known as Eldoria, there existed a grand kingdom surrounded by towering mountains and endless forests. 
The kingdom was ruled by King Arin, a wise and just ruler who had inherited the throne from his forefathers. Despite the tranquility within the palace walls, shadows loomed over the land as a dark sorcerer named Malakar plotted to seize control.

The king's daughter, Princess Elara, was a skilled warrior with a strong spirit, who vowed to defend Eldoria. Alongside her was Sir Cedric, a noble knight whose loyalty knew no bounds. Together with their trusted companions—a quick-witted thief named Finn, a gifted healer named Lyra, and a retired soldier called Garen—they formed a group known as the Guardians.

One day, the Guardians set out on a quest to recover an ancient artifact capable of vanquishing Malakar's magic. The journey took them through dense jungles, scorching deserts, and deep caves. They encountered fearsome beasts, cunning bandits, and dangerous traps laid by the sorcerer's minions.

After many trials, they finally stood before the artifact within the depths of a forgotten temple. However, Malakar's shadow loomed over them as he unleashed his dark magic. With swift movements and the strength of their bond, the Guardians fought fiercely against the sorcerer. With Princess Elara's decisive strike, they shattered his staff, banishing his power forever.

Peace returned to Eldoria as the Guardians were celebrated across the realm. Though their journey had ended, the bond forged between them remained unbreakable. They knew that as long as the darkness lurked beyond the mountains, the Guardians would be ready to defend their kingdom once more.
"""


factor   =  1500 / len(story.split()) 

epoch  =  int(len(story.split()) * factor)

print(f"Total Words  {len(story.split())}---- > ",epoch )


# Sample QA pairs
qa_pairs = [
    ("Who ruled the kingdom of Eldoria?", "King Arin"),
    ("What did Princess Elara vow to defend?", "Eldoria"),
    ("Who was the noble knight loyal to Princess Elara?", "Sir Cedric"),
    ("What did the Guardians set out to recover?", "artifact"),
    ("Who was the dark sorcerer plotting to seize control?", "Malakar"),
    ("What is the name of the healer in the Guardians?", "Lyra"),
    ("distant realm known as  ?","Eldoria")
]


# Create vocabulary
unique_words = set(word.lower().replace('?', '') for word in story.split())
for q, a in qa_pairs:
    unique_words.update(q.lower().replace('?', '').split())
    unique_words.add(a.lower())

vocab = {word: idx for idx, word in enumerate(unique_words, start=1)}
PAD_IDX = 0
vocab['<pad>'] = PAD_IDX

# Simple tokenizer function
def simple_tokenizer(text, vocab):
    tokens = text.lower().replace('?', '').split()
    return [vocab[token] for token in tokens if token in vocab]

# Tokenize the story
story_tokens = simple_tokenizer(story, vocab)

# Calculate the maximum length needed for any story + question combination
max_length = max(len(story_tokens) + len(simple_tokenizer(q, vocab)) for q, _ in qa_pairs)

def prepare_inputs_and_targets(qa_pairs, vocab, story_tokens, max_length):
    inputs = []
    targets = []

    for question, answer in qa_pairs:
        question_tokens = simple_tokenizer(question, vocab)
        combined_tokens = story_tokens + question_tokens
        combined_tokens = combined_tokens + [PAD_IDX] * (max_length - len(combined_tokens))
        inputs.append(combined_tokens)
        targets.append(vocab[answer.lower()])
    
    return torch.tensor(inputs), torch.tensor(targets)

# Prepare inputs and targets
inputs, targets = prepare_inputs_and_targets(qa_pairs, vocab, story_tokens, max_length)


# Model definition
class QA_Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super(QA_Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        x = self.embedding(x) * torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        x = x.transpose(0, 1)  # Switch to `[sequence_length, batch_size, feature_dim]`
        encoded = self.encoder(x)
        encoded = encoded.mean(dim=0)  # Pooling over the time dimension
        # encoded = torch.softmax(encoded, dim=1)
        return self.fc(encoded)

# Hyperparameters
vocab_size = len(vocab)
d_model = 32
num_heads = 4
num_layers = 2
num_epochs = epoch

# Initialize the model
model = QA_Transformer(vocab_size, d_model, num_heads, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    if epoch %  500 == 0:

        print("------------------------------------------------------")
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        model.eval()
        with torch.no_grad():
            output = model(inputs[5:])
            predicted_indices = output.argmax(dim=-1)
            for idx, pred_idx in enumerate(predicted_indices):
                predicted_answer = list(vocab.keys())[list(vocab.values()).index(pred_idx.item())]
                print(f"Question: {qa_pairs[idx][0]}, Predicted Answer: {predicted_answer}, Actual Answer: {qa_pairs[idx][1]}")
        print("------------------------------------------------------")
    elif epoch %100 ==0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Inference
model.eval()
with torch.no_grad():
    output = model(inputs)
    predicted_indices = output.argmax(dim=-1)
    for idx, pred_idx in enumerate(predicted_indices):
        predicted_answer = list(vocab.keys())[list(vocab.values()).index(pred_idx.item())]
        print(f"Question: {qa_pairs[idx][0]}, Predicted Answer: {predicted_answer}, Actual Answer: {qa_pairs[idx][1]}")
