
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from .prefetch_generator import BackgroundGenerator

class WFlowDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, num_data_processers=0, data_processer=None, data_processer_batch_size=1):
        super(WFlowDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)
        self.num_data_processers = num_data_processers
        self.data_processer = data_processer
        self.data_processer_batch_size = data_processer_batch_size

    def __iter__(self):
        if self.num_data_processers > 0 and self.data_processer is not None:
            return BackgroundGenerator(super().__iter__(), max_prefetch=self.num_data_processers, data_processer=self.data_processer, data_processer_batch_size=self.data_processer_batch_size)
        return super(WFlowDataLoader, self).__iter__()

    def __len__(self):
        return len(self.dataset)//self.data_processer_batch_size
