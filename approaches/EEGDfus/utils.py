import torch
import time
from torch.optim import Adam


def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"])


    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1500, gamma=.1, verbose=True
    )

    best_valid_loss = 1e10
    start_time = time.time()

    for epoch_no in range(config["epochs"]):
        epoch_start_time = time.time()
        avg_loss = 0
        count = 0
        model.train()

        for batch_no, (clean_batch, noisy_batch) in enumerate(train_loader, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            optimizer.zero_grad()

            loss = model(clean_batch, noisy_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            avg_loss += loss.item()
            count += 1

        epoch_time = time.time() - epoch_start_time
        avg_train_loss = avg_loss / count
        print(f"Epoch [{epoch_no + 1}/{config['epochs']}] - Train Loss: {avg_train_loss:.6f}, Time: {epoch_time:.2f}s")

        lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():

                for batch_no, (clean_batch, noisy_batch) in enumerate(valid_loader, start=1):
                    clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                    loss = model(clean_batch, noisy_batch)
                    avg_loss_valid += loss.item()

            avg_valid_loss = avg_loss_valid / batch_no
            if best_valid_loss > avg_valid_loss:
                best_valid_loss = avg_valid_loss
                print(f"*** New best validation loss: {avg_valid_loss:.6f} at epoch {epoch_no + 1} ***")

                if foldername != "":
                    torch.save(model.state_dict(), output_path)
            else:
                print(f"Validation Loss: {avg_valid_loss:.6f}")

    total_time = time.time() - start_time
    avg_time_per_epoch = total_time / config["epochs"]
    print(f"\nTraining completed!")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average time per epoch: {avg_time_per_epoch:.2f}s")
    
    if foldername != "":
        torch.save(model.state_dict(), final_path)