#!/usr/bin/env python3

from src.model.arch import SceneGraphGenerator
# from src.model.loss import 
import const
import torch
import sys


def train(model, optimizer, loss, data):
    training_loss = []
    interval = max(1, const.EPOCHS // 10)
    for epoch in range(const.EPOCHS):
        if not (epoch+1) % interval: print('-' * 10)
        training_loss.append([])

        for batch in data:
            optimizer.zero_grad()

            X, y = batch
            y_pred = model(X.to(device))

            batch_loss = loss(y_pred, y.to(device))
            batch_loss.backward()

            optimizer.step()

            training_loss[-1].append(batch_loss)
        if not (epoch+1) % interval: print(f'Epoch: {epoch+1}\tLoss: {sum(training_loss[-1]) / len(data)}')
    print('-' * 10)


if __name__ == '__main__':
    model = SceneGraphGenerator().to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM)
    loss = ()
    train(model, optimizer, loss, dataloader)
    torch.save(model, const.SAVE_MODEL_PATH / 'w2v.pt')

    embed()
