import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from csv import DictWriter


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def acc_func(y_true, y_pred):
    y_pred = torch.round(y_pred)
    return (y_true == y_pred).sum() / len(y_true)


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
    load_saved_model=False,
    # apply_l1=False,
    # apply_l2=False,
    # l1_weight=0.0001,
    # l2_weight=0.0001,
):
    if load_saved_model:
        print("Loading saved model")
        model.load_state_dict(torch.load(f"{(model._get_name())}_best_model.pth"))
    model = model.to(DEVICE)
    # acc_func = Accuracy(task='binary').to(DEVICE)
    train_epoch_loss = []
    test_epoch_loss = []

    train_epoch_acc = []
    test_epoch_acc = []

    try:
        for epoch in range(EPOCHS):
            epoch_loss = 0
            epoch_acc = 0
            for x, y in train_loader:
                model.train()
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                # y = y.unsqueeze(1).to(DEVICE)
                y_hat = model(x).squeeze()
                loss_value = loss(y_hat, y)
                optimizer.zero_grad()

                # TODO: #! Fix Regularization, Ask why it's not working?
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
                epoch_acc += acc_func(y, y_hat).item()
            train_epoch_loss.append(epoch_loss / len(train_loader))
            train_epoch_acc.append(epoch_acc / len(train_loader))

            if epoch % print_every == 0:
                print(f"Epoch {epoch} | Train Loss: {epoch_loss/len(train_loader):.5f}")
                print(
                    f"Epoch {epoch} | Train Acc: {epoch_acc*100/len(train_loader):.2f}%"
                )
                # print("--------------------------------------------------")

            with torch.no_grad():
                model.eval()
                epoch_loss = 0
                epoch_acc = 0
                for x, y in test_loader:
                    x = x.to(DEVICE)
                    # y = y.unsqueeze(1).to(DEVICE)
                    y = y.to(DEVICE)
                    y_hat = model(x).squeeze()
                    loss_value = loss(y_hat, y)
                    epoch_loss += loss_value.item()
                    epoch_acc += acc_func(y, y_hat).item()
                epoch_loss /= len(test_loader)

                if epoch % print_every == 0:
                    print(f"Epoch {epoch} | Test Loss: {epoch_loss:.5f}")
                    print(
                        f"Epoch {epoch} | Test Acc: {epoch_acc*100/len(test_loader):.2f}%"
                    )
                    try:
                        if epoch_loss < min(test_epoch_loss):
                            print(
                                f"Test loss decreased from {min(test_epoch_loss):.5f} to {epoch_loss:.5f} saving new best model"
                            )
                            torch.save(
                                model.state_dict(),
                                f"{(model._get_name())}_best_model.pth",
                            )
                    except ValueError:
                        print(
                            f"Test loss decreased from inf to {epoch_loss:.5f} saving new best model"
                        )
                        torch.save(
                            model.state_dict(), f"{(model._get_name())}_best_model.pth"
                        )
                    print("--------------------------------------------------")
                test_epoch_loss.append(epoch_loss)
                test_epoch_acc.append(epoch_acc / len(test_loader))
    except KeyboardInterrupt:
        history = {
            "loss": train_epoch_loss,
            "val_loss": test_epoch_loss,
            "accuracy": train_epoch_acc,
            "val_accuracy": test_epoch_acc,
        }
        with open(f"{(model._get_name())}_history.csv", 'a') as f_object:
            field_names = list(history.keys())
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            dictwriter_object.writerow(history)
            f_object.close()
            

        print("Interrupted, returning saved history")
        return history
    history = {
        "loss": train_epoch_loss,
        "val_loss": test_epoch_loss,
        "accuracy": train_epoch_acc,
        "val_accuracy": test_epoch_acc,
    }

    return history


# * Loading word2vec model
def Load_word2vec(path):
    import os
    import shutil
    import gensim

    # Provide the full path to the "word2vec_model" file

    # Check if the file exists
    if not os.path.exists(path):
        print(f"Error: The file '{path}' does not exist.")
    else:
        # Create a temporary directory to copy the model
        temp_dir = "C:\\Temp"  # Change this to a writable directory if needed

        try:
            # Create the temporary directory if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)

            # Copy the model to the temporary directory
            temp_model_path = os.path.join(temp_dir, "word2vec_model")
            shutil.copy(path, temp_model_path)

            # Load pre-trained Word2Vec model from the temporary directory
            model = gensim.models.Word2Vec.load(temp_model_path)
            print("Word2Vec model loaded successfully!")
            return model

        except PermissionError as e:
            print(f"Error: Permission denied while loading the model: {e}")
        except Exception as e:
            print(f"Error: An unexpected error occurred: {e}")
