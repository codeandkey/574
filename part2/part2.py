from args import args
import torch
from torch import nn, optim
import numpy as np
from mnist_loader import load_data
import math

EPOCHS = 150
BATCH = 64

train_data, _, test_data = load_data()

train_samples = []
train_labels = []
test_samples = []
test_labels = []

train_samples = np.array(train_data[0])
train_labels = np.array(train_data[1])
test_samples = np.array(test_data[0])
test_labels = np.array(test_data[1])

if args.activation == 'sigmoid':
    activation = nn.Sigmoid()
elif args.activation == 'relu':
    activation = nn.ReLU()
elif args.activation == 'tanh':
    activation = nn.Tanh()

model = nn.Sequential(
    nn.Linear(784, args.units1),
    activation,
    nn.Linear(args.units1, args.units2),
    nn.Softmax(1)
)

if args.weight_init == 'zero':
    with torch.no_grad():
        for value in model.parameters():
            value.fill_(0)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay,
                      momentum=0.9)

indices = np.arange(len(train_samples))

def acc(samples, labels):
    pred_labels = model(torch.tensor(samples)).argmax(axis=1).detach().numpy()
    return np.sum(pred_labels == labels.astype(int)) / len(labels)

train_acc_history = []
test_acc_history = []

for epoch in range(EPOCHS):
    # Generate batches every epoch to improve training stability
    np.random.shuffle(indices)
    batches = np.array_split(indices, int(math.ceil(len(train_samples) / BATCH)))

    for batch in batches:
        # Reset model gradients
        optimizer.zero_grad()

        # Compute loss function
        pred = model(torch.tensor(train_samples[batch]))
        loss = criterion(pred, torch.tensor(train_labels[batch]))

        # Optimize model
        loss.backward()
        optimizer.step()

    train_acc = acc(train_samples, train_labels)
    test_acc = acc(test_samples, test_labels)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f'Epoch {epoch} / {EPOCHS}: loss {loss.item():.2f}, train acc {train_acc}, test acc {test_acc}')

with open('results.txt', 'w')  as f:
    f.write(f'[{train_acc_history}, {test_acc_history}]')
