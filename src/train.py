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
from metrics.accuracy import accuracy
from metrics.rouge import evalROUGE
from metrics.logger import MetricsLogger



def writeHistoryHeader(history_path):
    with open(history_path, "w") as f:
        f.write("epoch;train_loss;train_acc;val_loss;val_acc;val_r1_p;val_r1_r;val_r1_f1;val_r2_p;val_r2_r;val_r2_f1;val_rl_p;val_rl_r;val_rl_f1\n")

def writeHistoryEntry(history_path, epoch, train_metrics, val_metrics):
    with open(history_path, "a") as f:
        train_avgs = train_metrics.averages()
        val_avgs = val_metrics.averages()

        f.write(
            f"{epoch};{train_avgs['loss']};{train_avgs['accuracy']};{val_avgs['loss']};{val_avgs['accuracy']};" +
            f"{val_avgs['rouge1']['precision']};{val_avgs['rouge1']['recall']};{val_avgs['rouge1']['fmeasure']};" +
            f"{val_avgs['rouge2']['precision']};{val_avgs['rouge2']['recall']};{val_avgs['rouge2']['fmeasure']};" +
            f"{val_avgs['rougeL']['precision']};{val_avgs['rougeL']['recall']};{val_avgs['rougeL']['fmeasure']}"
        )
        f.write("\n")


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
        epochs : int
            Number of epochs to train.
        starting_epoch : int
            Number from where the epoch count will start.
        history_path : str
            CSV file where the training history will be stored.
        checkpoints_path : str
            Directory where the checkpoints will be stored.
        checkpoints_frequency : int
            Number of epochs after which a checkpoint will be created.
        checkpoint_best : bool
            If True, a checkpoint will be created each time a model has a better validation accuracy.
        model_name : str
            Name of the pretrained model (e.g. bert-base-uncased)
"""
def train(model, loss, optimizer, scheduler, train_dataloader, val_dataloader, epochs, device, tokenizer,
          history_path, checkpoints_path, checkpoints_frequency, checkpoint_best, model_name, starting_epoch=1):
    def _createCheckpoint(path, epoch_num, model, model_name, optimizer, metrics):
        torch.save({
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }, path)

    epochs = epochs + starting_epoch - 1
    curr_best_val_accurary = -1
    train_metrics = MetricsLogger()
    val_metrics = MetricsLogger()

    for epoch_num in range(starting_epoch, epochs+1):
        train_metrics.reset()
        val_metrics.reset()

        # Training
        model.train()
        for documents, labels in tqdm(train_dataloader, desc=f"Epoch {epoch_num}/{epochs}"):
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model.predict(documents, device)
            batch_loss = loss(outputs, labels.float())
            batch_loss.backward()
            optimizer.step()

            train_metrics.add("loss", batch_loss.item())
            train_metrics.add("accuracy", accuracy(labels, outputs))
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for documents, labels in tqdm(val_dataloader, desc=f"Validation"):
                labels = labels.to(device)

                outputs = model.predict(documents, device)
                batch_loss = loss(outputs, labels.float())

                val_metrics.add("loss", batch_loss.item())
                val_metrics.add("accuracy", accuracy(labels, outputs))
                val_metrics.add("rouge", evalROUGE(model, tokenizer, documents, labels, outputs))

        print(f"Train: {train_metrics.format(['loss', 'accuracy'])}")
        print(f"Val: {val_metrics.format(['loss', 'accuracy', 'rouge'])}")


        # Checkpoints
        is_best_model = (val_metrics.averages()["accuracy"] > curr_best_val_accurary and checkpoint_best)
        is_final_epoch = (epoch_num == epochs)
        if epoch_num % checkpoints_frequency == 0 or is_best_model or is_final_epoch:
            checkpoint_path = os.path.join(checkpoints_path, f"cp_{model_name.replace('/', '_')}_ep{epoch_num}.tar")
            print(f"Saving checkpoint at {checkpoint_path}")
            _createCheckpoint(checkpoint_path, epoch_num, model, model_name, optimizer, val_metrics.averages())

            if is_best_model:
                curr_best_val_accurary = val_metrics.averages()["accuracy"]
                with open(os.path.join(checkpoints_path, f"best.txt"), "w") as f:
                    f.write(
                        f"Epoch {epoch_num} | " +
                        f"{val_metrics.format(['loss', 'accuracy', 'rouge'])}"
                    )

        writeHistoryEntry(history_path, epoch_num, train_metrics, val_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--model", type=str, required=True, help="Model to use as starting point (e.g. bert-base-uncased)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a preprocesses dataset")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up steps")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--checkpoint-best", action="store_true", help="Create a checkpoint for the best model")
    parser.add_argument("--checkpoints-path", type=str, default="./__checkpoints__", help="Directory where the checkpoints will be stored")
    parser.add_argument("--history-path", type=str, default="./__checkpoints__/history.csv", help="CSV file where the training history will be stored")
    parser.add_argument("--checkpoints-freq", type=int, default=100, help="Number of epochs after which a checkpoint will be created")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(mode=True)
    if torch.cuda.is_available(): os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTSummarizer(args.model).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model)
    datasets = load_dataset(args.dataset, tokenizer=tokenizer, splits=["train", "validation"])
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

    writeHistoryHeader(args.history_path)

    print("-- Starting training --")
    train(
        model = model,
        model_name = args.model,
        loss = loss, 
        optimizer = optimizer, 
        scheduler = scheduler,
        train_dataloader = train_dataloader, 
        val_dataloader = val_dataloader, 
        tokenizer = tokenizer,
        epochs = args.epochs,
        starting_epoch = starting_epoch,
        device = device,
        history_path = args.history_path,
        checkpoints_path = args.checkpoints_path,
        checkpoints_frequency = args.checkpoints_freq,
        checkpoint_best = args.checkpoint_best
    )
