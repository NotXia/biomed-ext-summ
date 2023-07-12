from models.loader import loadModel
from data.loader import loadDataset
import torch
from tqdm import tqdm
import numpy as np
import argparse
import os
import random
from metrics.recall import recall
from metrics.rouge import evalROUGE
from metrics.logger import MetricsLogger
from models.LongformerSummarizer import LongformerSummarizer



def writeHistoryHeader(history_path=None):
    if history_path == None: return 

    with open(history_path, "w") as f:
        f.write("epoch;train_loss;train_recall;val_loss;val_recall;val_r1_p;val_r1_r;val_r1_f1;val_r2_p;val_r2_r;val_r2_f1;val_rl_p;val_rl_r;val_rl_f1\n")

def writeHistoryEntry(epoch, train_metrics, val_metrics, history_path=None):
    if history_path is None: return 

    with open(history_path, "a") as f:
        train_avgs = train_metrics.averages()
        val_avgs = val_metrics.averages()

        f.write(
            f"{epoch};{train_avgs['loss']};{train_avgs['recall']};{val_avgs['loss']};{val_avgs['recall']};" +
            f"{val_avgs['rouge1']['precision']};{val_avgs['rouge1']['recall']};{val_avgs['rouge1']['fmeasure']};" +
            f"{val_avgs['rouge2']['precision']};{val_avgs['rouge2']['recall']};{val_avgs['rouge2']['fmeasure']};" +
            f"{val_avgs['rougeL']['precision']};{val_avgs['rougeL']['recall']};{val_avgs['rougeL']['fmeasure']}"
        )
        f.write("\n")


"""
    Computes the average loss of each document of the batch ignoring paddings
"""
def perSentenceLoss(loss, batch_predictions, batch_labels, batch_num_sentences):
    total_loss = 0.0

    for batch_i in range(len(batch_predictions)):
        predictions = batch_predictions[batch_i]
        labels = batch_labels[batch_i]
        num_sentences = batch_num_sentences[batch_i]

        predictions = predictions[:num_sentences]
        labels = labels[:num_sentences]

        total_loss += loss(predictions, labels)

    return total_loss / len(batch_predictions)


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
        history_path : str
            CSV file where the training history will be stored.
        checkpoint : str
            Path to the checkpoint to load
        checkpoints_path : str
            Directory where the checkpoints will be stored.
        checkpoints_frequency : int
            Number of epochs after which a checkpoint will be created.
        checkpoint_best : bool
            If True, a checkpoint will be created each time a model has a better validation recall.
        accumulation_steps : int
            Gradient accumulation steps.
"""
def train(model, loss, optimizer, train_dataloader, val_dataloader, epochs, device,
          history_path, checkpoint, checkpoints_path, checkpoints_frequency, checkpoint_best, accumulation_steps=1, use_mixed_precision=False):
    def _createCheckpoint(path, epoch_num, model, optimizer, metrics):
        torch.save({
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "model_name": model.model_name,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }, path)

    model = model.to(device)
    loss = loss.to(device)
    starting_epoch = 1
    curr_best_val_recall = -1
    train_metrics = MetricsLogger()
    val_metrics = MetricsLogger()
    scaler = torch.cuda.amp.GradScaler()

    if checkpoint is not None:
        print(f"-- Loading checkpoint at {checkpoint} --")
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1

    if not os.path.exists(checkpoints_path): os.makedirs(checkpoints_path)
    writeHistoryHeader(history_path)

    epochs = epochs + starting_epoch - 1
    for epoch_num in range(starting_epoch, epochs+1):
        train_metrics.reset()
        val_metrics.reset()

        # Training
        model.train()
        optimizer.zero_grad()
        for i, (documents, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch_num}/{epochs}")):
            labels = labels.float().to(device)
            predictions = None

            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    predictions, logits = model(documents)
                    batch_loss = perSentenceLoss(loss, logits, labels, documents["num_sentences"])
                    acc_loss = batch_loss / accumulation_steps
                scaler.scale(acc_loss).backward()
            else:
                predictions, logits = model(documents)
                batch_loss = perSentenceLoss(loss, logits, labels, documents["num_sentences"])
                acc_loss = batch_loss / accumulation_steps
                acc_loss.backward()

            if ((i+1) % accumulation_steps == 0) or ((i+1) == len(train_dataloader)):
                if use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_metrics.add("loss", batch_loss.item())
            train_metrics.add("recall", recall(labels, predictions))

        # Validation
        model.eval()
        with torch.no_grad():
            for documents, labels in tqdm(val_dataloader, desc=f"Validation"):
                labels = labels.float().to(device)

                outputs, logits = model(documents)
                batch_loss = perSentenceLoss(loss, logits, labels, documents["num_sentences"])

                # Creates summaries for ROUGE
                ext_summaries = []
                for i in range(len(labels)): # For each element of batch
                    ext_summary_size = len( (labels[i] == 1).nonzero(as_tuple=True)[0] )
                    ext_sentences, _ = model.summarizeFromDataset(outputs[i], documents["ids"][i], ext_summary_size)
                    ext_summaries.append( "\n".join(ext_sentences) )
                    
                val_metrics.add("loss", batch_loss.item())
                val_metrics.add("recall", recall(labels, outputs))
                val_metrics.add("rouge", evalROUGE(documents["ref_summary"], ext_summaries))

        print(f"Train: {train_metrics.format(['loss', 'recall'])}")
        print(f"Val: {val_metrics.format(['loss', 'recall', 'rouge'])}")


        # Checkpoints
        is_best_model = (val_metrics.averages()["recall"] > curr_best_val_recall and checkpoint_best)
        is_final_epoch = (epoch_num == epochs)
        if (epoch_num % checkpoints_frequency == 0) or is_best_model or is_final_epoch:
            checkpoint_path = os.path.join(checkpoints_path, f"cp_{model.model_name.replace('/', '_')}_ep{epoch_num:03d}.tar")
            print(f"Saving checkpoint at {checkpoint_path}")
            _createCheckpoint(checkpoint_path, epoch_num, model, optimizer, val_metrics.averages())

            if is_best_model:
                curr_best_val_recall = val_metrics.averages()["recall"]
                with open(os.path.join(checkpoints_path, f"best.txt"), "w") as f:
                    f.write(
                        f"Epoch {epoch_num} | " +
                        f"{val_metrics.format(['loss', 'recall', 'rouge'])}"
                    )

        writeHistoryEntry(epoch_num, train_metrics, val_metrics, history_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--model", type=str, required=True, help="Model to use as starting point (e.g. bert-base-uncased)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a preprocesses dataset")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--checkpoint-best", action="store_true", help="Create a checkpoint for the best model")
    parser.add_argument("--checkpoints-path", type=str, default="./__checkpoints__", help="Directory where the checkpoints will be stored")
    parser.add_argument("--history-path", type=str, default="./__checkpoints__/history.csv", help="CSV file where the training history will be stored")
    parser.add_argument("--checkpoints-freq", type=int, default=100, help="Number of epochs after which a checkpoint will be created")
    parser.add_argument("--mixed-precision", action="store_true", default=False, help="Use mixed precision (FP32 + FP16)")
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(mode=True)
    if torch.cuda.is_available(): os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loadModel(args.model)
    datasets = loadDataset(args.dataset, model.tokenizer, splits=["train", "validation"])
    train_dataloader = torch.utils.data.DataLoader(datasets["train"], batch_size=args.batch_size)
    val_dataloader = torch.utils.data.DataLoader(datasets["validation"], batch_size=args.batch_size)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    if isinstance(model, LongformerSummarizer):
        torch.use_deterministic_algorithms(mode=False)

    print("-- Starting training --")
    train(
        model = model,
        loss = loss, 
        optimizer = optimizer, 
        train_dataloader = train_dataloader, 
        val_dataloader = val_dataloader, 
        accumulation_steps = args.accum_steps,
        epochs = args.epochs,
        device = device,
        history_path = args.history_path,
        checkpoint = args.checkpoint,
        checkpoints_path = args.checkpoints_path,
        checkpoints_frequency = args.checkpoints_freq,
        checkpoint_best = args.checkpoint_best,
        use_mixed_precision = args.mixed_precision
    )
