#!/usr/bin/env python3

from ..data.visualgenome import get_generators
from .arch import SceneGraphGenerator
from .loss import GraphLoss
from .. import const
import torch


def fit(model, optimizer, loss, train):
    training_loss = []
    for epoch in range(const.EPOCHS):
        training_loss.append([])

        for batch in train:
            optimizer.zero_grad()

            X, y = batch
            X = X.unsqueeze(0)
            y_pred = model(X.to(const.DEVICE))
            y = [y,]

            batch_loss = loss(y, y_pred)
            sum(map(lambda weight, loss: weight * loss, const.LOSS_ALPHAS.values(), batch_loss)).backward()

            optimizer.step()

            training_loss[-1].append(batch_loss)
    print('-' * 10)


if __name__ == '__main__':
    train, val, test = get_generators()
    model = SceneGraphGenerator().to(const.DEVICE)
    model.backbone.eval()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM)
    loss = GraphLoss()

    fit(model, optimizer, loss, train)
    torch.save(model, const.MODEL_DIR / 'model.pt')
