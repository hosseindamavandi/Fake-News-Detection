import torch
from torch.utils.data import Dataset, DataLoader


def hello():
    print("Hello Hussein!")


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
    # apply_l1=False,
    # apply_l2=False,
    # l1_weight=0.0001,
    # l2_weight=0.0001,
):
    def acc_func(y_true, y_pred):
        y_pred = torch.round(y_pred)
        return (y_true == y_pred).sum() / len(y_true)

    model = model.to(DEVICE)
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
            
            # #* Regularization L1 and L2
            # if apply_l1 and apply_l2:
            #     parameters = []
            #     for parameter in model.parameters():
            #         parameters.append(parameter.view(-1))
            #     l1 = l1_weight * model.compute_l1_loss(torch.cat(parameters))
            #     l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))
            #     loss_value += l1
            #     loss_value += l2
                
            # #* Regularization L1 only
            # elif apply_l1:
            #     parameters = []
            #     for parameter in model.parameters():
            #         parameters.append(parameter.view(-1))
            #     l1 = l1_weight * model.compute_l1_loss(torch.cat(parameters))
            #     loss_value += l1
            
            # #* Regularization L2 only
            # if apply_l1 and apply_l2:
            #     parameters = []
            #     for parameter in model.parameters():
            #         parameters.append(parameter.view(-1))
            #     l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))
            #     loss_value += l2

            loss_value.backward()
            optimizer.step()

            epoch_loss += loss_value.item()
            epoch_acc += acc_func(y, y_hat)
        train_epoch_loss.append(epoch_loss / len(train_loader))
        train_epoch_acc.append(epoch_acc / len(train_loader))

        if epoch % print_every == 0:
            print(f"Epoch {epoch} | Train Loss: {epoch_loss/len(train_loader)}")
            print(f"Epoch {epoch} | Train Acc: {epoch_acc/len(train_loader)}")
            # print("--------------------------------------------------")

        with torch.no_grad():
            model.eval()
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
                print(f'Test loss decreased from {min(test_epoch_loss)} to {epoch_loss} saving new best model')
                torch.save(model.state_dict(), "best_model.pth")
            if epoch % print_every == 0:
                print(f"Epoch {epoch} | Test Loss: {epoch_loss/len(test_loader)}")
                print(f"Epoch {epoch} | Test Acc: {epoch_acc/len(test_loader)}")
                print("--------------------------------------------------")


    return train_epoch_loss, test_epoch_loss, train_epoch_acc, test_epoch_acc
