import time
import numpy as np
import torch
import traceback
from model import vgg16bn_unet

class Train:
    def __init__(self, model, dataloader, cuda, opt, loss, epochs):
        self.model = model
        self.train_dataloader = dataloader
        self.cuda = cuda
        self.optimizer = opt
        self.loss = loss
        self.n_epochs = epochs

    def train_one_epoch(self):
        t_epoch_start = time.time()

        self.model.train()

        losses = []

        print("train batches: %s" % (len(self.train_dataloader)))
        for batch_i, (features_batch, labels_batch) in enumerate(self.train_dataloader):
            print(f"batch_i={batch_i}")
            t0 = time.time()

            try:
                if self.cuda:
                    features_batch = features_batch.cuda()
                    labels_batch = labels_batch.cuda().long()

                # forward
                out_features = self.model(features_batch)
                loss = loss(out_features, labels_batch)

                if loss == 0 or not torch.isfinite(loss):
                    continue

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # for print
                losses.append(loss.item())

                time_span = time.time() - t0

                print(
                    "batch {:05d} | Time(s):{:.2f}s | Loss:{:.4f}".format(batch_i, time_span, loss.item()))

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        durations = time.time() - t_epoch_start

        return {
            "train_durations": durations,
            "average_train_loss": np.mean(losses),
        }

    def train(self):

        # ================================================
        # train loop
        # ================================================
        print("total epochs: %s" % (self.n_epochs))

        try:
            for epoch_i in range(self.n_epochs):
                print()
                print("------------------------")
                print("Epoch {:05d} training...".format(epoch_i))
                print("------------------------")

                self.epoch_i = epoch_i

                one_epoch_result = self.train_one_epoch()
                self.scheduler.step()

                print(
                    "Epoch {:05d} training complete...: | Time(s):{:.2f}s | Average Loss:{:.4f}".format(
                        epoch_i, one_epoch_result["train_durations"], one_epoch_result["average_train_loss"]
                    ))

                # ================================================
                # after each epoch ends
                # ================================================

                # val_one_epoch_result = self.val_one_epoch()
                # if val_one_epoch_result["average_val_loss"] < self.best_loss:
                #     self.best_loss = val_one_epoch_result["average_val_loss"]
                #     self.save_model()


        except KeyboardInterrupt:
            pass


    def save_model(self):
        print("saving model ...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, './checkpoint/best_parameters.pth')

if __name__ == "main":
    

    
