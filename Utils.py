import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def split_data(data, ratio=0.8, seed=42):
    vectors = data[:, :-1]
    labels = data[:, -1]
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        vectors, labels, train_size=ratio, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def Training(
    model,
    train_loader,
    test_loader,
    EPOCHS,
    DEVICE,
    loss,
    optimizer,
    print_every=10,
):
    def acc_func(y_true, y_pred):
        y_pred = torch.round(y_pred)
        return (y_true == y_pred).sum() / len(y_true)

    train_epoch_loss = []
    test_epoch_loss = []

    train_epoch_acc = []
    test_epoch_acc = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.unsqueeze(1).to(DEVICE)
            y_hat = model(x)
            loss_value = loss(y_hat, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            epoch_loss += loss_value.item()
            epoch_acc += acc_func(y, y_hat)
        train_epoch_loss.append(epoch_loss / len(train_loader))
        train_epoch_acc.append(epoch_acc / len(train_loader))

        if epoch % print_every == 0:
            print(f"Epoch {epoch} | Train Loss: {epoch_loss/len(train_loader)}")
            print(f"Epoch {epoch} | Train Acc: {epoch_acc/len(train_loader)}")

        with torch.no_grad():
            epoch_loss = 0
            epoch_acc = 0
            for x, y in test_loader:
                x = x.to(DEVICE)
                y = y.unsqueeze(1).to(DEVICE)
                y_hat = model(x)
                loss_value = loss(y_hat, y)
                epoch_loss += loss_value.item()
                epoch_acc += acc_func(y, y_hat)
            test_epoch_loss.append(epoch_loss / len(test_loader))
            test_epoch_acc.append(epoch_acc / len(test_loader))
            if epoch_loss < min(test_epoch_loss):
                torch.save(model.state_dict(), "model.pth")
            if epoch % print_every == 0:
                print(f"Epoch {epoch} | Test Loss: {epoch_loss/len(test_loader)}")
                print(f"Epoch {epoch} | Test Acc: {epoch_acc/len(test_loader)}")

    return train_epoch_loss, test_epoch_loss, train_epoch_acc, test_epoch_acc