import torch


class Agent:
    def __init__(self,
                 name,
                 model,
                 optimizer,
                 criterion,
                 lr_scheduler,
                 train_loader,
                 test_loader,
                 stats,
                 train_freq=None,
                 stat_freq=None,
                 ):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.train_iter = iter(self.train_loader)
        self.test_loader = test_loader
        self.stats = stats
        self._buffer_stat = {
            'train_loss': 0.0,
            'train_total': 0,
            'train_correct': 0,
        }

        self.epoch_size = len(train_loader)
        self.stat_freq = self.epoch_size if stat_freq is None else stat_freq
        self.train_freq = 1 if train_freq is None else train_freq

        self.global_iteration = 0
        self.iteration = 0
        self.consensus_params = dict()

    def test(self):
        self.model.eval()
        self.model.training = False
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            acc = 100. * correct / total

        self.stats.add('val_precision', acc)
        self.stats.add('val_loss', test_loss)

    def make_iteration(self):
        # scheduled training and saving statistics
        self.global_iteration += 1

        if self.global_iteration % self.train_freq == 0:
            if self.iteration % self.epoch_size == 0:
                self.train_iter = iter(self.train_loader)

            self.iteration += 1
            train_stats = self._fit_batch()

            self._buffer_stat['train_loss'] += train_stats['train_loss']
            self._buffer_stat['train_total'] += train_stats['train_total']
            self._buffer_stat['train_correct'] += train_stats['train_correct']

        if self.global_iteration % self.stat_freq == 0:
            self.stats.add('train_loss', self._buffer_stat['train_loss'])
            self._buffer_stat['train_loss'] = 0.0

            acc = 100. * self._buffer_stat['train_correct'] / self._buffer_stat['train_total']
            self.stats.add('train_precision', acc)
            self._buffer_stat['train_correct'] = 0
            self._buffer_stat['train_total'] = 0

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.stats.add('lr', lr)
            self.stats.dump_to_file()

        self.lr_scheduler.step()

    def _fit_batch(self):
        # fit self.model on one batch
        self.model.train()

        inputs, targets = self.train_iter.next()
        inputs, targets = inputs.cuda(), targets.cuda()

        self.optimizer.zero_grad()
        outputs = self.model(inputs)  # Forward Propagation
        loss = self.criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        self.optimizer.step()  # Optimizer update

        _, predicted = torch.max(outputs.data, 1)

        return {'train_loss': loss.item(),
                'train_total': targets.size(0),
                'train_correct': predicted.eq(targets.data).cpu().sum()}

    def load_flatten_params_to_model(self, params):
        used_params = 0
        for p in self.model.parameters():
            cnt_params = p.numel()
            p.data.copy_(torch.Tensor(params[used_params:used_params + cnt_params]).view(p.shape).to(p.dtype))
            used_params += cnt_params

    def get_flatten_params(self):
        return torch.cat(
            [p.data.to(torch.float32).view(-1) for p in self.model.parameters()]).detach().clone().cpu().numpy()

    def set_consensus_params(self, agent_name, params):
        self.consensus_params[agent_name] = params

    def get_params_for_averaging(self):
        return self.consensus_params
