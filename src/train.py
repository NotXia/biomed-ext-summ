from models.bert_summ import BERTSummarizer
from data.dataset_loader import load_dataset
from transformers import BertTokenizer
import torch
from tqdm import tqdm
import numpy as np
import argparse
import os
import random
from utilities.transformer import NoamScheduler



def accuracy(labels, predictions):
    accuracies = []

    for i in range(len(labels)): # Batch handling
        summary_sentences_idxs = (labels[i] == 1).nonzero(as_tuple=True)[0]
        selected_sentences_idxs = sorted(torch.topk(predictions[i], len(summary_sentences_idxs)).indices)

        idx_ref, idx_sel = 0, 0
        correct_choices = 0
        while idx_ref < len(summary_sentences_idxs) and idx_sel < len(selected_sentences_idxs):
            if summary_sentences_idxs[idx_ref] == selected_sentences_idxs[idx_sel]:
                correct_choices += 1
                idx_ref += 1
                idx_sel += 1
            elif summary_sentences_idxs[idx_ref] > selected_sentences_idxs[idx_sel]:
                idx_sel += 1
            else:
                idx_ref += 1

        accuracies.append( correct_choices / len(summary_sentences_idxs) )

    return np.average(accuracies)



"""
    Makes a prediction for a BERT-based model
"""
def getBERTModelsOutput(model, document_batch):
    ids = document_batch["ids"].to(device)
    segments_ids = document_batch["segments_ids"].to(device)
    clss_mask = document_batch["clss_mask"].to(device)
    bert_mask = document_batch["bert_mask"].to(device)
    return model(ids, segments_ids, clss_mask, bert_mask)


"""
    Training loop.

    Parameters
    ----------
        model : Model
        loss : Loss
        optimizer : Optimizer
        train_dataloader : DataLoader
        val_dataloader : DataLoader
        device : Device
        getModelOutput : (model, batch) -> Tensor
            Function that makes a prediction with the model on a batch of documents
        epochs : int
            Number of epochs to train.
        starting_epoch : int
            Number from where the epoch count will start.
        checkpoints_path : str
            Directory where the checkpoints will be stored.
        checkpoints_frequency : int
            Number of epochs after which a checkpoint will be created.
        checkpoint_best : bool
            If True, a checkpoint will be created each time a model has a better validation accuracy.
"""
def train(model, getModelOutput, loss, optimizer, scheduler, train_dataloader, val_dataloader, epochs, device, checkpoints_path, checkpoints_frequency, checkpoint_best, starting_epoch=1):
    def _createCheckpoint(path, epoch_num, model, optimizer, avg_loss, avg_accuracy):
        torch.save({
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_loss,
            "val_accuracy": avg_accuracy,
        }, path)

    epochs = epochs + starting_epoch - 1
    curr_best_val_accurary = -1
    total_train_loss, total_train_accuracy = 0, 0
    total_val_loss, total_val_accuracy = 0, 0
    avg_train_loss, avg_train_accurary = 0, 0
    avg_val_loss, avg_val_accurary = 0, 0

    for epoch_num in range(starting_epoch, epochs+1):
        total_train_loss, total_train_accuracy = 0, 0
        total_val_loss, total_val_accuracy = 0, 0
        avg_train_loss, avg_train_accurary = 0, 0
        avg_val_loss, avg_val_accurary = 0, 0

        # Training
        model.train()
        for documents, labels in tqdm(train_dataloader, desc=f"Epoch {epoch_num}/{epochs}"):
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = getModelOutput(model, documents)
            batch_loss = loss(outputs, labels.float())
            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()
            total_train_accuracy += accuracy(labels, outputs)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for documents, labels in val_dataloader:
                labels = labels.to(device)

                outputs = getModelOutput(model, documents)
                batch_loss = loss(outputs, labels.float())

                total_val_loss += batch_loss.item()
                total_val_accuracy += accuracy(labels, outputs)

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accurary = total_train_accuracy / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accurary = total_val_accuracy / len(val_dataloader)

        print( 
            f"Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f} | " +
            f"Train acc: {avg_train_accurary:.6f} | Val acc: {avg_val_accurary:.6f}"
        )

        # Checkpoints
        if epoch_num % checkpoints_frequency == 0:
            print("Saving checkpoint")
            _createCheckpoint(
                os.path.join(checkpoints_path, f"cp_e{epoch_num}.tar"), 
                epoch_num, model, optimizer, avg_val_loss, avg_val_accurary
            )

        if avg_val_accurary > curr_best_val_accurary and checkpoint_best:
            print("Saving checkpoint for best model")
            _createCheckpoint(
                os.path.join(checkpoints_path, f"cp_best.tar"), 
                epoch_num, model, optimizer, avg_val_loss, avg_val_accurary
            )

            with open(os.path.join(checkpoints_path, f"best.txt"), "w") as f:
                f.write(
                    f"Epoch {epoch_num} | " +
                    f"Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f} | " +
                    f"Train acc: {avg_train_accurary:.6f} | Val acc: {avg_val_accurary:.6f}"
                )

    print("Saving final checkpoint")
    _createCheckpoint(
        os.path.join(checkpoints_path, f"cp_e{epoch_num}.tar"), 
        epoch_num, model, optimizer, avg_val_loss, avg_val_accurary
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--model", type=str, required=True, help="Model to use as starting point (e.g. bert-base-uncased)")
    parser.add_argument("--model-family", type=str, choices=["bert"], required=True, help="Family of the model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a dataset parsed with preprocess.py")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warm-up steps")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--checkpoint-best", action="store_true", help="Create a checkpoint for the best model")
    parser.add_argument("--checkpoints-path", type=str, default="./__checkpoints__", help="Directory where the checkpoints will be stored")
    parser.add_argument("--checkpoints-freq", type=int, default=100, help="Number of epochs after which a checkpoint will be created")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(mode=True)
    if torch.cuda.is_available(): os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTSummarizer(args.model).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True)
    datasets = load_dataset(args.dataset, args.model_family, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(datasets["train"], batch_size=args.batch_size)
    val_dataloader = torch.utils.data.DataLoader(datasets["validation"], batch_size=args.batch_size)
    loss = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = NoamScheduler(optimizer, args.warmup, model.scheduler_d_model)
    starting_epoch = 1

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)

    if args.checkpoint is not None:
        print(f"-- Loading checkpoint at {args.checkpoint} --")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1
        scheduler.epoch = checkpoint["epoch"]

    print("-- Starting training --")
    train(
        model = model,
        getModelOutput = getBERTModelsOutput,
        loss = loss, 
        optimizer = optimizer, 
        scheduler = scheduler,
        train_dataloader = train_dataloader, 
        val_dataloader = val_dataloader, 
        epochs = args.epochs,
        starting_epoch = starting_epoch,
        device = device,
        checkpoints_path = args.checkpoints_path,
        checkpoints_frequency = args.checkpoints_freq,
        checkpoint_best = args.checkpoint_best
    )
