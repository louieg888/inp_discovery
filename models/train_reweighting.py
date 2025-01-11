from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import os
import sys
import toml
import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from dataset.utils import get_dataloader, setup_dataloaders
from models.models import INP, CriticModel, SyntheticBernoulliReweightingModel, VectorRepresentationModel
from models.loss import ELBOLoss

EVAL_ITER = 50
SAVE_ITER = 500
MAX_EVAL_IT = 50

def acc_func(unnormed_vector_for_two_classes, target):
    # we will predict with  Y | r(X) only.
    y_pred = torch.max(unnormed_vector_for_two_classes.view(-1, 2), dim=1)[1].view(-1)
    return (y_pred == target.view(-1)).float().view(-1)

class Trainer:
    def __init__(self, config, save_dir, load_path=None, last_save_it=0):
        self.config = config
        self.last_save_it = last_save_it

        self.device = config.device
        self.train_dataloader, self.critic_dataloader, self.val_dataloader, self.id_val_dataloader, _, extras = setup_dataloaders(config)

        for k, v in extras.items():
            config.__dict__[k] = v

        self.num_epochs = config.num_epochs

        self.model = INP(config)
        self.z_representation_model = VectorRepresentationModel(input_size=1, hidden_size=16, num_hidden_layers=1, output_size=1)
        self.x_representation_model = VectorRepresentationModel(input_size=2, hidden_size=16, num_hidden_layers=1, output_size=1)
        self.critic_model = CriticModel(input_size=3, hidden_size=16, num_hidden_layers=2, output_size=2)

        self.model.to(self.device)

        self.models = {
            "pred_model": self.model,
            "x_rep_model": self.x_representation_model,
            "z_rep_model": self.z_representation_model,
            "critic_model": self.critic_model,
        }

        self.pred_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.critic_optimizer = torch.optim.Adam( list(self.critic_model.parameters()) + list(self.z_representation_model.parameters()), lr=self.config.lr)
        self.x_representation_optimizer = torch.optim.Adam(self.x_representation_model.parameters(), lr=self.config.lr)

        self.optimizers = {
            "pred_optim": self.pred_optimizer,
            "critic_optim": self.critic_optimizer,
            "x_rep_optim": self.x_representation_optimizer,
        }
        
        self.loss_func = ELBOLoss(beta=config.beta)
        if load_path is not None:
            print(f"Loading model from state dict {load_path}")
            state_dict = torch.load(load_path)
            self.model.load_state_dict(state_dict, strict=False)
            loaded_states = set(state_dict.keys())

        own_trainable_states = []
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                own_trainable_states.append(name)

        if load_path is not None:
            own_trainable_states = set(own_trainable_states)
            print(f"\n States not loaded from state dict:")
            print(
                *sorted(list(own_trainable_states.difference(loaded_states))), sep="\n"
            )
            print(f"Unknown states:")
            print(
                *sorted(
                    list(loaded_states.difference(set(self.model.state_dict().keys())))
                ),
                sep="\n",
            )

        self.save_dir = save_dir

        
    def get_loss(self, x_context, y_context, x_target, y_target, knowledge):
        if self.config.sort_context:
            x_context, indices = torch.sort(x_context, dim=1)
            y_context = torch.gather(y_context, 1, indices)

        #todo: only for rep
        if self.config.reweight:
            x_context, x_target = self.x_representation_model(x_context), self.x_representation_model(x_target)
        
        if self.config.use_knowledge:
            output = self.model(
                x_context,
                y_context,
                x_target,
                y_target=y_target,
                knowledge=knowledge,
            )
        else:
            output = self.model(
                x_context, y_context, x_target, y_target=y_target, knowledge=None
            )

        # when you get here, figure out how to get accuracy in.
        loss, kl, negative_ll = self.loss_func(output, y_target)

        results = {
            'loss': loss,
            'kl' : kl,
            'negative_ll': negative_ll
        }

        return results

    def run_batch_train(self, batch):
        context, target, knowledge, ids = batch
        x_context, y_context = context
        x_target, y_target = target
        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        results = self.get_loss(
            x_context, y_context, x_target, y_target, knowledge
        )

        return results

    def run_reweighting_batch(self, batch):
        """
        In a single batch of retraining, we look at y_context and try and create an estimator p(y|z) for the relationship. 
        """
        context, target, knowledge, ids = batch
        x_context, y_context = context
        x_target, y_target = target
        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        #todo: add another part of the dataloader to extract spurious correlations.
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        y_hat = self.reweighting_model(ids['extras']['z'])
        acc_vec = acc_func(y_hat, y_target)
        loss_vec = ce_loss(y_hat.view(-1, 2), y_target.long().view(-1)).view(-1)

        results = {
            "loss": torch.mean(loss_vec), 
            "acc": torch.mean(acc_vec),
        }

        print(results)
    
        return results

    def run_batch_eval(self, batch, num_context=5):
        context, target, knowledge, ids = batch
        x_target, y_target = target
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        context_idx = np.random.choice(x_target.shape[1], num_context, replace=False)

        x_context, y_context = x_target[:, context_idx, :], y_target[:, context_idx, :]

        results = self.get_loss(
            x_context, y_context, x_target, y_target, knowledge
        )

        return results

    def set_critic_training(self):
        self.critic_model.train()
        self.z_representation_model.train()

        self.x_representation_model.eval()
        self.model.eval()
    
    def set_pred_training(self):
        self.critic_model.eval()
        self.z_representation_model.eval()

        self.x_representation_model.train()
        self.model.train()

    #todo: rename appropriately
    def train_reweighting_and_weight_train_set(self):
        def _generate_folds(dataset, config):
            # Create (train, val) splits
            kf = KFold(n_splits=5, shuffle=False)

            folds = []
            fold_indices = []
            
            # generate k folds of data and save in folds
            for train_idx, val_idx in kf.split(range(len(dataset))):
                fold_indices.append((train_idx, val_idx))
                train_subset = torch.utils.data.Subset(dataset, train_idx)
                val_subset = torch.utils.data.Subset(dataset, val_idx)
                
                original_batch_size, config.batch_size = config.batch_size, 1000
                train_dataloader = get_dataloader(train_subset, config)
                val_dataloader = get_dataloader(val_subset, config)
                config.batch_size = original_batch_size

                folds.append((train_dataloader, val_dataloader))

        dataset = self.train_dataloader.dataset
        folds = _generate_folds(dataset, self.config)

        reweighting_models = []

        for (train_dataloader, val_dataloader) in folds:
            self.reweighting_model = SyntheticBernoulliReweightingModel()
            self.reweighting_model.to(self.device)
            self.reweighting_pred_optimizer = torch.optim.Adam(self.reweighting_model.parameters(), lr=1e-2)

            for epoch in range(101):
            # for epoch in range(101):
                for batch in train_dataloader:
                    self.reweighting_model.train()
                    self.reweighting_pred_optimizer.zero_grad()
                    reweighting_results = self.run_reweighting_batch(batch)
                    acc = reweighting_results['acc']
                    loss = reweighting_results['loss']
                    print(f"epoch: {epoch}, acc: {acc}, loss: {loss}")

                    loss.retain_grad()
                    loss.backward()
                    self.reweighting_pred_optimizer.step()
                    # wandb.log({"reweighting_train_loss": loss})
                    # wandb.log({"reweighting_train_acc": acc})

                    losses, val_loss = self.eval_weighting(val_dataloader)
                    mean_eval_loss = np.mean(list(losses.values()))
                    # wandb.log({"reweighting_mean_eval_loss": mean_eval_loss})
                    # wandb.log({"reweighting_eval_loss": val_loss})
                    for k, v in losses.items():
                        if k == "loss": 
                            continue

                        # wandb.log({f"reweighting_eval_loss_{k}": v})

                    if val_loss < min_eval_loss:
                        min_eval_loss = val_loss
                        torch.save(
                            self.reweighting_model.state_dict(), f"{self.save_dir}/reweighting_model_best.pt"
                        )
                        torch.save(
                            self.reweighting_pred_optimizer.state_dict(),
                            f"{self.save_dir}/reweighting_optim_best.pt" 
                        )
                        print(f"Best model saved at iteration {self.last_save_it + it}")

                    it += 1

            reweighting_models.append({
                "model": self.reweighting_model,
                "min_eval_loss": min_eval_loss,
                "val_dataloader": val_dataloader
            })

        index_weight_mappings = []
        # for (reweighted_model, validation_dataloader): 
        for model_and_extras in reweighting_models:
            model = model_and_extras['model']
            val_dataloader = model_and_extras['val_dataloader']

            for batch in val_dataloader: 
                context, target, knowledge, ids = batch
                x_context, y_context = context
                x_target, y_target = target
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)
                y_target = y_target.to(self.device)

                # get_probabilities(reweighted_model, validation_loader)
                ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
                y_hat = model(knowledge)
                acc_vec = acc_func(y_hat, y_target)
                loss_vec = ce_loss(y_hat.view(-1, 2), y_target.long().view(-1)).view(-1)

                # map from cross entropy to probabilities
                y_z_prob_vec = (loss_vec * -1).exp()
                y_z_prob_vec.view(y_target.shape)

                index_weight_mappings.append({
                    "indices": val_dataloader.dataset.indices,
                    "y_unnormalized_weights": 1 / y_z_prob_vec
                })

        # all of this is just to account for potential randomization / shuffling in splits
        # so that the weights go to the right place  
        all_indices = torch.cat([torch.tensor(entry["indices"]) for entry in index_weight_mappings])
        all_weights = torch.cat([entry["y_unnormalized_weights"] for entry in index_weight_mappings])

        sorted_indices, sort_order = torch.sort(all_indices)
        sorted_weights = all_weights.view(sorted_indices.shape[0], -1)[sort_order]

        torch.save(sorted_weights, "saves/weights.pt")

        self.train_dataloader.dataset.add_weights(sorted_weights)

        # makes sure we don't do the weighting twice if the train and critic are using the same dataset object
        if not self.critic_dataloader.dataset.weighted:
            self.critic_dataloader.dataset.add_weights(sorted_weights)

        # todo: assert that full range has been reached at the end of it all. 

    def run_critic_training_step(self):
        self.set_critic_training()
        try: 
            inner_batch = next(self.critic_dataloader_iterator)
        except StopIteration: 
            self.critic_dataloader_iterator = iter(self.critic_dataloader)
            inner_batch = next(self.critic_dataloader_iterator)

        self.critic_optimizer.zero_grad()

        losses = self.get_all_losses(inner_batch)
        critic_loss = losses['critic_loss'] 

        wandb.log({"critic_loss": critic_loss.item()})
        # Backpropagation and optimizer step
        critic_loss.backward()
        self.critic_optimizer.step()

    def train(self):
        it = 0
        min_eval_loss = np.inf

        if self.config.reweight: 
            self.train_reweighting_and_weight_train_set()

        # todo: make sure you account for p(y) before you leave the synthetic dataset. 
        # this is technically not necessary since it's constant due to weight normalization and upsampling
        # create the weighted dataloader with the modified representation
        # use the modified dataloader to do the rest of the INP process

        # once this works, we need to adapt the INP to learn the representation on the fly. How will this work?

        it = 0
        min_eval_loss = np.inf

        if self.config.reweight:
            self.critic_dataloader_iterator = iter(self.critic_dataloader)

        for epoch in range(self.num_epochs + 1):
            #self.scheduler.step()

            # this is the outer training loop
            for batch in self.train_dataloader:
                # don't forget to set models apprpriately to eval and stuff
                self.set_pred_training()

                self.pred_optimizer.zero_grad()
                
                if self.config.reweight:
                    self.x_representation_optimizer.zero_grad()

                results = self.get_all_losses(batch)
                loss = results['loss']
                kl = results['kl']
                negative_ll = results['negative_ll']

                if self.config.reweight:
                    info_loss = results['information_loss']
                    total_loss = loss + info_loss
                else:
                    total_loss = loss


                total_loss.backward()
                self.pred_optimizer.step()

                if self.config.reweight:
                    self.x_representation_optimizer.step()

                wandb.log({"train_loss": loss})
                wandb.log({"train_negative_ll": negative_ll})
                wandb.log({"train_kl": kl})

                if self.config.reweight:
                    wandb.log({"info_loss": info_loss})

                if it % EVAL_ITER == 0 and it > 0:
                    for val_name, val_set in zip(("ood", "id"), (self.val_dataloader, self.id_val_dataloader)):
                        losses, val_loss = self.eval(val_set)
                        mean_eval_loss = np.mean(list(losses.values()))
                        wandb.log({f"{val_name}_mean_eval_loss": mean_eval_loss})
                        wandb.log({f"{val_name}_eval_loss": val_loss})
                        for k, v in losses.items():
                            wandb.log({f"{val_name}_eval_loss_{k}": v})

                        if val_name == "id" and val_loss < min_eval_loss and it > 1500:

                            min_eval_loss = val_loss
                            for model_name, model in self.models.items():
                                torch.save(model.state_dict(), f"{self.save_dir}/{model_name}_best.pt")
                            for optim_name, optim in self.optimizers.items():
                                torch.save(optim.state_dict(), f"{self.save_dir}/{optim_name}_best.pt")
                            
                            print(f"Best model saved at iteration {self.last_save_it + it}")


                # only run critic training if we are doing nurd adaptation
                if self.config.reweight: 
                    self.set_critic_training()
                    for _ in range(8):
                        self.run_critic_training_step()

                it += 1

        return min_eval_loss

    def get_all_losses(self, batch):
        pred_losses = self.run_batch_train(batch)

        context, target, knowledge, ids = batch
        x_context, y_context = context
        x_target, y_target = target
        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        x, y, z = x_target, y_target, ids['z']

        # representations
        rep_x = self.x_representation_model(x)
        rep_z = self.z_representation_model(z)
        shuffled_rep_z = rep_z[:, torch.randperm(rep_z.size(1))]
        # critic model predictions
        logit_real_given_xy_real_z = self.critic_model(torch.cat((rep_x, y, rep_z), axis=2)).view(-1, 2)
        logit_real_given_xy_fake_z = self.critic_model(torch.cat((rep_x, y, shuffled_rep_z), axis=2)).view(-1, 2)

        # Labels for real and fake data
        # note: correct probability logit is therefore in index 1, ie [0 1]
        joint_labels = torch.ones([logit_real_given_xy_real_z.size(0), ], dtype=torch.long).to(self.device)
        marginal_labels = torch.zeros([logit_real_given_xy_fake_z.size(0), ], dtype=torch.long).to(self.device)

        # Losses for real and fake data
        loss_real = F.cross_entropy(logit_real_given_xy_real_z, joint_labels)
        loss_fake = F.cross_entropy(logit_real_given_xy_fake_z, marginal_labels)

        # Combine losses
        # todo: add in critic loss when you get here.
        critic_loss = 0.5 * (loss_real + loss_fake)
        critic_real_preds = torch.softmax(logit_real_given_xy_real_z, dim=1).argmax(axis=1)
        critic_fake_preds = torch.softmax(logit_real_given_xy_fake_z, dim=1).argmax(axis=1)
        critic_real_acc = (critic_real_preds == joint_labels).sum() / joint_labels.size(0)
        critic_fake_acc = (critic_fake_preds == marginal_labels).sum() / marginal_labels.size(0)
        critic_acc = (critic_real_acc + critic_fake_acc) / 2.

        # Information loss
        log_p_real_given_xy_real_z = torch.log_softmax(logit_real_given_xy_real_z, dim=1)[:, 1]
        log_p_fake_given_xy_real_z = torch.log_softmax(logit_real_given_xy_real_z, dim=1)[:, 0]
        information_loss = log_p_real_given_xy_real_z - log_p_fake_given_xy_real_z

        losses = {
            "critic_loss": critic_loss.sum(),
            "critic_acc": critic_acc,
            "information_loss": torch.abs(information_loss.sum()),
            **pred_losses
        }

        return losses


    def eval(self, val_dataloader):
        print('Evaluating')
        it = 0
        self.model.eval()
        with torch.no_grad():
            loss_num_context = [3, 5, 10]
            if self.config.min_num_context == 0:
                loss_num_context = [0] + loss_num_context
            losses_dict = dict(zip(loss_num_context, [[] for _ in loss_num_context]))

            val_losses = []
            for batch in val_dataloader:
                for num_context in loss_num_context:
                    results = self.run_batch_eval(batch, num_context=num_context)
                    loss = results['loss']
                    val_results = self.run_batch_train(batch)
                    val_loss = val_results['loss']
                    losses_dict[num_context].append(loss.to("cpu").item())
                    val_losses.append(val_loss.to("cpu").item())
                    
                it += 1
                if it > MAX_EVAL_IT:
                    break
            losses_dict = {k: np.mean(v) for k, v in losses_dict.items()}
            val_loss = np.mean(val_losses)



            # loss_num_context = [3, 5, 10]
            # if self.config.min_num_context == 0:
            #     loss_num_context = [0] + loss_num_context
            # losses_dict = dict(zip(loss_num_context, [[] for _ in loss_num_context]))

            # val_losses = []
            # for batch in self.val_dataloader:
            #     for num_context in loss_num_context:
            #         results = self.run_batch_eval(batch, num_context=num_context)
            #         loss = results['loss']
            #         val_results = self.run_batch_train(batch)
            #         val_loss = val_results['loss']
            #         losses_dict[num_context].append(loss.to("cpu").item())
            #         val_losses.append(val_loss.to("cpu").item())
                    
            #     it += 1
            #     if it > MAX_EVAL_IT:
            #         break
            # losses_dict = {k: np.mean(v) for k, v in losses_dict.items()}
            # val_loss = np.mean(val_losses)
            
        return losses_dict, val_loss

    def eval_weighting(self, val_dataloader):
        print('Evaluating')
        it = 0
        self.reweighting_model.eval()
        with torch.no_grad():
            losses_dict = {
                "acc": [],
                "loss": []
            }

            val_losses = []
            
            # need to implement k-fold cross validation here 
            for batch in val_dataloader:
                reweighting_results = self.run_reweighting_batch(batch)

                val_acc = reweighting_results['acc']
                val_loss = reweighting_results['loss']

                losses_dict['acc'].append(val_acc.to("cpu").item())
                losses_dict['loss'].append(val_loss.to("cpu").item())
                val_losses.append(val_loss.to("cpu").item())
                    
                it += 1
                if it > MAX_EVAL_IT:
                    break
            losses_dict = {k: np.mean(v) for k, v in losses_dict.items()}
            val_loss = np.mean(val_losses)
            
        return losses_dict, val_loss



def get_device():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda:{}".format(0))
    else:
        device = "cpu"
    print("Using device: {}".format(device))
    return device


def meta_train(trial, config, run_name_prefix="run"):
    device = get_device()
    config.device = device

    # Create save folder and save config
    save_dir = f"./saves/{config.project_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_no = len(os.listdir(save_dir))
    save_no = [
        int(x.split("_")[-1])
        for x in os.listdir(save_dir)
        if x.startswith(run_name_prefix)
    ]
    if len(save_no) > 0:
        save_no = max(save_no) + 1
    else:
        save_no = 0
    save_dir = f"{save_dir}/{run_name_prefix}_{save_no}"
    os.makedirs(save_dir, exist_ok=True)
    
    trainer = Trainer(config=config, save_dir=save_dir)

    config = trainer.config

    # save config
    config.write_config(f"{save_dir}/config.toml")

    wandb.init(
        project=config.project_name, name=f"{run_name_prefix}_{save_no}", config=vars(config)
    )
    best_eval_loss = trainer.train()
    wandb.finish()

    return best_eval_loss


if __name__ == "__main__":
    # resume_training('run_7')
    import random
    import numpy as np
    from config import Config

    # read config from config.toml
    config = toml.load(f"config.toml")
    config = Config(**config)

    # set seed
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # begin study
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda x: meta_train(
            x, 
            config=config, 
            run_name_prefix=config.run_name_prefix
        ),
        n_trials=config.n_trials,
    )
