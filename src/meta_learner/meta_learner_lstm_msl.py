import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
import  numpy as np
from    copy import deepcopy
from src.models.learner import Learner, Net

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.restore_path = args.restore_path


        self.net = Net(self.n_way)

        self.loss = 0
        self.best_loss = 0
        self.log_path = args.log_path

        if(self.restore_path is not None):
            checkpoint = torch.load(self.restore_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.loss = checkpoint['loss']

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

        self.low_dict = self.net.state_dict()

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        This Corresponds to the Meta-Learning Stage
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # List of Losses and Accs on the Query Set across Tasks
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)] 
        weights = [0.1*i for i in range(self.update_step + 1)]

        # Copy of the local net
        # net = deepcopy(self.net)

        optim = torch.optim.SGD(self.net.parameters(), lr=self.update_lr)

        for i in range(task_num):
            
            

            # Compute Loss on Support Set
            logits = self.net(x_spt[i])
            loss = F.cross_entropy(logits, y_spt[i])

            # Calculate Loss and Acc on Query Set before Updating
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i])
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # Calculate Loss and Acc on Query Set after Updating
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i])
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i])
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                # 3. theta_pi = theta_pi - train_lr * grad
                optim.zero_grad()
                loss.backward()
                optim.step()
                # 4. evaluation on query set
                logits_q = self.net(x_qry[i])
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # META-LEARNER OPTIMIZATION
        # Sum over all losses on query set across all tasks
        # Simple version of multi stel loss for maml 
        # loss_q = losses_q[-1] / task_num
        loss_q = (sum([losses_q[i]*weights[i] for i in range(len(losses_q))])/sum(weights))/task_num
        
        self.net.load_state_dict(self.low_dict)

        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        self.low_dict = self.net.state_dict()
        
        # Saving checkpoint
        self.loss = loss_q 
        
        # if first best_loss (=0) then receive the first loss
        if(self.best_loss == 0):
            self.best_loss = self.loss
        
        if(self.loss < self.best_loss):
            print('New best loss')
            self.best_loss = self.loss   

            torch.save({
            'model_state_dict': self.net.state_dict(),
            'loss': self.loss,
            'best_loss': self.best_loss,
            }, self.log_path + f"/best_model.pth")

        # Accuracy Calculation
        accs = np.array(corrects) / (querysz * task_num)
        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        This corresponds to the fine-tuning stage
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 3
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        optim = torch.optim.SGD(net.parameters(), lr=self.update_lr)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = self.net(x_qry)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi

            optim.zero_grad()
            loss.backward()
            optim.step()

            logits_q = net(x_qry)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net
        accs = np.array(corrects) / querysz
        return accs

def main():
    pass

if __name__ == '__main__':
    main()