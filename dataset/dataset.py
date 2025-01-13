import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

ROOT = '/Users/louiemcconnell/CHUV/code/informed-meta-learning-31FF/data'

class SetKnowledgeTrendingSinusoids(Dataset):
    def __init__(
            self, split='train', 
            root=f'{ROOT}/trending-sinusoids', 
            knowledge_type='full', 
            split_file='splits'
        ):
        self.data = pd.read_csv(f'{root}/data.csv')
        self.knowledge = pd.read_csv(f'{root}/knowledge.csv')
        self.value_cols = [c for c in self.data.columns if c.isnumeric()]
        self.dim_x = 1
        self.dim_y = 1
        if split_file is None:
            split_file = 'splits'
        self.train_test_val_split = pd.read_csv(f'{root}/{split_file}.csv')
        self.split = split
        self.knowledge_type = knowledge_type
        self.knowledge_input_dim = 4

        self._split_data()

    def _split_data(self):
        if self.split == 'train':
            train_ids = self.train_test_val_split[self.train_test_val_split['split'] == 'train'].curve_id
            self.data = self.data[self.data.curve_id.isin(train_ids)]
        elif self.split == 'val' or self.split == 'valid':
            val_ids = self.train_test_val_split[self.train_test_val_split['split'] == 'val'].curve_id
            self.data = self.data[self.data.curve_id.isin(val_ids)]
        elif self.split == 'test':
            test_ids = self.train_test_val_split[self.train_test_val_split['split']== 'test'].curve_id
            self.data = self.data[self.data.curve_id.isin(test_ids)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        y = self.data.iloc[idx, :][self.value_cols].values
        x = np.linspace(-2, 2, len(y))
        curve_id = self.data.iloc[idx]['curve_id']
        
        knowledge = self.get_knowledge(curve_id)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1) #[bs,  num_points, x_size]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) #[bs,  num_points, x_size]

        return x, y, knowledge


    def get_knowledge(self, curve_id):
        knowledge = self.knowledge[self.knowledge.curve_id == curve_id].drop('curve_id', axis=1).values
        knowledge = torch.tensor(knowledge, dtype=torch.float32).reshape(3, 1)
        indicator = torch.eye(3)
        knowledge = torch.cat([indicator, knowledge], dim=1)

        if self.knowledge_type == 'abc':
            revealed = np.random.choice([0, 1, 2])
            mask = torch.zeros((3, 1))
            mask[revealed] = 1.0
            knowledge = knowledge * mask
        elif self.knowledge_type == 'abc2':
            mask = torch.zeros((3, 1))
            num_revealed = np.random.choice([1, 2])
            revealed = np.random.choice([0, 1, 2], num_revealed, replace=False)
            mask[revealed] = 1.0
            knowledge = knowledge * mask
        elif self.knowledge_type == 'a':
            knowledge = knowledge[0, :].unsqueeze(0)
        elif self.knowledge_type == 'b':
            knowledge = knowledge[1, :].unsqueeze(0)
        elif self.knowledge_type == 'c':
            knowledge = knowledge[2, :].unsqueeze(0)
        elif self.knowledge_type == 'full':
            pass
        elif self.knowledge_type  == 'none':
            knowledge = torch.zeros_like(knowledge)
        else:
            raise NotImplementedError

        return knowledge


class SetKnowledgeTrendingSinusoidsDistShift(SetKnowledgeTrendingSinusoids):
    def __init__(self, split='train', root='./data/trending-sinusoids-dist-shift', knowledge_type='full', split_file='splits'):
        super().__init__(split=split, root=root, knowledge_type=knowledge_type, split_file=split_file)


   
class Temperatures(Dataset):
    def __init__(self, split='train', root='./data/temperatures', knowledge_type='min_max'):
        region = 'AK'
        self.data = pd.read_csv(f'{root}/2021-2022_{region}.csv')
        self.splits = pd.read_csv(f'{root}/2021-2022_{region}_splits.csv')
        if knowledge_type == 'desc':
            self.knowledge_df = pd.read_csv(f'{root}/2021-2022_{region}_gpt_descriptions.csv')    
        elif knowledge_type in ['min_max', 'min_max_month']:
            self.knowledge_df = pd.read_csv(f'{root}/2021-2022_{region}_knowledge.csv')
        elif knowledge_type == 'llama_embed':
            knowledge_ds = load_from_disk(f'{root}/2021-2022_{region}_desc-embeded-llama')
            self.knowledge_df = knowledge_ds.to_pandas()
        
        
        self.knowledge_type = knowledge_type
        if knowledge_type == 'min_max':
            self.knowledge_input_dim = 2
        elif knowledge_type == 'min_max_month':
            self.knowledge_input_dim = 3
        elif knowledge_type == 'desc':
            self.knowledge_input_dim = None
        elif knowledge_type =='llama_embed':
            self.knowledge_input_dim = 4096
        else:
            raise NotImplementedError

        self.split = split
        if self.split == 'train':
            dates = self.splits[self.splits.split == 'train'].LST_DATE
        elif self.split == 'val' or self.split == 'valid':
            dates = self.splits[self.splits.split == 'val'].LST_DATE
        elif self.split == 'test':
            dates = self.splits[self.splits.split == 'test'].LST_DATE
        
        self.data = self.data[self.data.LST_DATE.isin(dates)]

        self.dim_x = 1
        self.dim_y = 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        y = self.data.iloc[idx, 1:].values
        lst_date = self.data.iloc[idx, 0]   
        k_idx = self.knowledge_df[self.knowledge_df.LST_DATE == lst_date].index[0]
        x = np.linspace(-2, 2, len(y))
        if self.knowledge_type == 'min_max':
            knowledge = self.knowledge_df[['min', 'max']].iloc[k_idx, :].values
            knowledge = torch.tensor(knowledge, dtype=torch.float32)
        elif self.knowledge_type == 'min_max_month':
            knowledge = self.knowledge_df[['min', 'max', 'month']].iloc[k_idx, :].values
            knowledge = torch.tensor(knowledge, dtype=torch.float32)
        elif self.knowledge_type == 'desc':
            knowledge = self.knowledge_df.iloc[k_idx, :].description
        elif self.knowledge_type == 'llama_embed':
            knowledge = self.knowledge_df.iloc[k_idx, :].embed[0]
            knowledge = torch.tensor(knowledge)
        else:
            raise NotImplementedError
        

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return x, y, knowledge

def generate_synthetic_data(a, m=5000, b=0, c=1):
    y = torch.randint(low=0, high=2, size=(m, )).long()

    # z_g_y_probs = rho*torch.ones(y.shape)
    # z_g_y_probs[y==1] = rho
    # z_g_y_probs[y==0] = 1 - rho
    # z = torch.bernoulli(z_g_y_probs)
    z = torch.randn((m, )) + a*(2*y - 1)

    y_polar = 2*y - 1
    # z_polar = 2*z - 1

    eps_1 = torch.randn((m, ))
    eps_2 = torch.randn((m, ))

    # assert y.shape == z_polar.shape
    assert y.shape == eps_2.shape

    # x = torch.cat([(radius*cos_angle).view(-1,1), (radius*sin_angle).view(-1,1)], dim=1)
    # x= torch.cat([(y - 0.5*z + 0.7*eps_1).view(-1,1,), (y - z + 0.7*eps_2).view(-1,1,)], dim=1)
    x = torch.cat([(b + c * y - z + 3*eps_1).view(-1, 1,),
                  (b + c * y + z + 0.1*eps_2).view(-1, 1,)], dim=1)

    return x, y, z


class NuRD(Dataset):
    def __init__(self, split='train', root='./data/nurd', knowledge_type='min_max', from_csv=False, task="simple"):
        # add config here
        self.knowledge_type = knowledge_type 

        if from_csv: 
            self.data = pd.read_csv(f'{root}/nurd.csv', converters={"x": lambda x: list(eval(x))})
            self.data = self.data[self.data['label'] == split]
        else: 
            n_tasks = 1000

            # if task == "multi": 
            #     self.data_generating_params = (torch.rand(n_tasks, 2) - 1) * 20
            # else: 
            #     self.data_generating_params = torch.tensor([0, 1]).repeat(n_tasks,1)

            # N(4, 1)
            normal_random = torch.randn(n_tasks, 2) + 4
            binary_random = (torch.rand(n_tasks, 2) > 0.5).float() * 2 - 1
            normal_binary_random = normal_random * binary_random
            normal_binary_random[:, 1] = 1

            self.data_generating_params = normal_binary_random

            # self.data_generating_params = torch.cat( (torch.tensor([2, 1]).repeat(int(n_tasks / 2), 1), torch.tensor([-2, 1]).repeat(n_tasks - int(n_tasks / 2), 1)), axis=0)
            # self.data_generating_params = self.data_generating_params[torch.randperm(self.data_generating_params.size()[0])]


            all_data = []
            if split == 'train': 
                for i in range(n_tasks):
                    b, c = self.data_generating_params[i]
                    all_data.append(generate_synthetic_data(0.5, m=100, b=b, c=1))
            elif split == 'id_val':
                for i in range(20):
                    b, c = self.data_generating_params[i]
                    all_data.append(generate_synthetic_data(0.5, m=100, b=b, c=1))
            else:
                for i in range(20):
                    b, c = self.data_generating_params[i]
                    all_data.append(generate_synthetic_data(-0.9, m=100, b=b, c=1))

            x, y, z = [torch.cat([data[k] for data in all_data]) for k in range(3)]

            self.data = pd.DataFrame({"x": [list(el.numpy()) for el in x], "y": y, "z": z})
            self.data['task'] = np.floor(self.data.index / 100).astype(int)

        #todo: fix this so that the dimension is "real" (pre representation)
        self.dim_x = 2
        self.dim_y = 1
        self.knowledge_input_dim = 3
        self.split = split

        self.weighted = False
        self.use_optimal_rep = False
        self.upsample_factor = 10

        # z_min = self.data['x'][0].min(), self.data['x'][1].min()
        # z_max = self.data['x'][0].min(), self.data['x'][1].min()
        # x_ax = self.data['x'][0].min(), self.data['x'][1].min()

    def __len__(self):
        return len(self.data['task'].unique())
    
    def __getitem__(self, idx):
        indices = self.data.index[self.data['task'] == idx]

        if self.weighted:
            weights = self.weights[idx].flatten()
            weighted_dist = torch.distributions.Categorical(weights)
            num_samples = len(indices) * self.upsample_factor
            indices = weighted_dist.sample((num_samples,))
        elif self.split == 'train': # we only do weighting / upsampling on train, so this will make comparison easier
            indices = indices.repeat(self.upsample_factor)

        x = self.data.iloc[indices]['x'].values
        y = self.data.iloc[indices]['y'].values
        z = self.data.iloc[indices]['z'].values

        x = torch.tensor(list(x), dtype=torch.float32)
        y = torch.tensor(list(y), dtype=torch.float32).unsqueeze(-1)
        z = torch.tensor(list(z), dtype=torch.float32).unsqueeze(-1)

        if self.knowledge_type == "z":
            knowledge = z
        else:
            knowledge = self.get_knowledge(idx)

        if self.use_optimal_rep: 
            x = x.sum(axis=-1).unsqueeze(-1)

        return x, y, knowledge, z

    def set_use_optimal_rep(self):
        self.use_optimal_rep = True
        self.dim_x = 1
        
    def add_weights(self, weights): 
        if self.weighted: 
            raise Exception("This dataset already has a set of weights.")

        self.weighted = True
        self.weights = weights
        self.upsample_factor = 10   
        # self.set_use_optimal_rep()

    
    def get_knowledge(self, task): 
        knowledge = self.data_generating_params[task]
        knowledge = knowledge.view(2, 1)
        indicator = torch.eye(2)
        knowledge = torch.cat([indicator, knowledge], dim=1)

        if self.knowledge_type == 'bc':
            revealed = np.random.choice([0, 1])
            mask = torch.zeros((2, 1))
            mask[revealed] = 1.0
            knowledge = knowledge * mask
        elif self.knowledge_type == 'b':
            knowledge = knowledge[0, :].unsqueeze(0)
        elif self.knowledge_type == 'c':
            knowledge = knowledge[1, :].unsqueeze(0)
        elif self.knowledge_type == 'full':
            pass
        elif self.knowledge_type  == 'none':
            knowledge = torch.zeros_like(knowledge)
        else:
            raise NotImplementedError

        return knowledge



# class NuRD(Dataset):
#     def __init__(self, split='train', root='./data/nurd', knowledge_type='min_max', from_csv=False, task="rando"):
#         # add config here
#         if from_csv: 
#             self.data = pd.read_csv(f'{root}/nurd.csv', converters={"x": lambda x: list(eval(x))})
#             self.data = self.data[self.data['label'] == split]
#         else: 
#             if split == 'train': 
#                 data = generate_synthetic_data(0.5, m=10000)
#             elif split == 'id_val':
#                 data = generate_synthetic_data(0.5, m=2000)
#             else:
#                 data = generate_synthetic_data(-0.9, m=2000)

#             x, y, z = data
#             self.data = pd.DataFrame({"x": [list(el.numpy()) for el in x], "y": y, "z": z})
#             self.data['task'] = np.floor(self.data.index / 100).astype(int)

#         #todo: fix this so that the dimension is "real" (pre representation)
#         self.dim_x = 2
#         self.dim_y = 1
#         self.knowledge_input_dim = 1
#         self.split = split

#         self.weighted = False
#         self.use_optimal_rep = False
#         self.upsample_factor = 10

#         # z_min = self.data['x'][0].min(), self.data['x'][1].min()
#         # z_max = self.data['x'][0].min(), self.data['x'][1].min()
#         # x_ax = self.data['x'][0].min(), self.data['x'][1].min()

#     def __len__(self):
#         return len(self.data['task'].unique())
    
#     def __getitem__(self, idx):
#         indices = self.data.index[self.data['task'] == idx]

#         if self.weighted:
#             weights = self.weights[idx].flatten()
#             weighted_dist = torch.distributions.Categorical(weights)
#             num_samples = len(indices) * self.upsample_factor
#             indices = weighted_dist.sample((num_samples,))
#         elif self.split == 'train': # we only do weighting / upsampling on train, so this will make comparison easier
#             indices = indices.repeat(self.upsample_factor)

#         x = self.data.iloc[indices]['x'].values
#         y = self.data.iloc[indices]['y'].values
#         knowledge = self.data.iloc[indices]['z'].values

#         x = torch.tensor(list(x), dtype=torch.float32)
#         y = torch.tensor(list(y), dtype=torch.float32).unsqueeze(-1)
#         knowledge = torch.tensor(list(knowledge), dtype=torch.float32).unsqueeze(-1)

#         if self.use_optimal_rep: 
#             x = x.sum(axis=-1).unsqueeze(-1)

#         return x, y, knowledge, knowledge

#     def set_use_optimal_rep(self):
#         self.use_optimal_rep = True
#         self.dim_x = 1
        
#     def add_weights(self, weights): 
#         if self.weighted: 
#             raise Exception("This dataset already has a set of weights.")

#         self.weighted = True
#         self.weights = weights
#         self.upsample_factor = 10   
#         # self.set_use_optimal_rep()




class NuRD_(Dataset):
    def __init__(self, split='train', root='./data/nurd', knowledge_type='z', from_csv=False, task="single"):

        """
        Task types: 
            - single: the traditional NuRD task. Meta-Learning over a nuisance family of distributions.
            - multi: the NP task. Data is generated via )b + cy + z, small_var) and (b + cy - z, large_var)
        Knowledge type: 
            - z: add in knowledge of the z variables. 
            - "none": no knowledge. 
            - "b": just b. 
            - "c": just c. 
            - "bc": both.
        """
        self.knowledge_type = knowledge_type

        # add config here
        if from_csv: 
            self.data = pd.read_csv(f'{root}/nurd.csv', converters={"x": lambda x: list(eval(x))})
            self.data = self.data[self.data['label'] == split]
        else: 
            if task == "multi": 
                self.data_generating_params = (torch.rand(100, 2) - 1) * 2
            else: 
                self.data_generating_params = torch.tensor([0, 1]).repeat(100,1)

            tasks = []
            
            if split == 'train':
                for i in range(100): 
                    data = generate_synthetic_data(0.5, m=100, b=b, c=c)
                    tasks.append((b, c, data))
            elif split == "id_val":
                for i in range(20): 
                    data = generate_synthetic_data(0.5, m=100, b=b, c=c)
                    tasks.append((b, c, data))
            else: 
                for i in range(20): 
                    data = generate_synthetic_data(-0.9, m=100, b=b, c=c)
                    tasks.append((b, c, data))

            dfs = []
            for task_idx, task in enumerate(tasks): 
                b, c, (x, y, z) = task
                df = pd.DataFrame({"x": [list(el.numpy()) for el in x], "y": y, "z": z, })
                df['task'] = task_idx
                dfs.append(df)
                # self.data['task'] = np.floor(self.data.index / 100).astype(int)
            self.data = pd.concat(dfs)

        self.knowledge_input_dim = 1
        # if knowledge_type == "z":
        #     self.knowledge_input_dim = 1
        # else:
        #     self.knowledge_input_dim = 3

        #todo: fix this so that the dimension is "real" (pre representation)
        self.dim_x = 2
        self.dim_y = 1
        self.split = split

        self.weighted = False
        self.use_optimal_rep = False
        self.upsample_factor = 10

        # z_min = self.data['x'][0].min(), self.data['x'][1].min()
        # z_max = self.data['x'][0].min(), self.data['x'][1].min()
        # x_ax = self.data['x'][0].min(), self.data['x'][1].min()

    def __len__(self):
        return len(self.data['task'].unique())
    
    def __getitem__(self, idx):
        indices = self.data.index[self.data['task'] == idx]

        if self.weighted:
            weights = self.weights[idx].flatten()
            weighted_dist = torch.distributions.Categorical(weights)
            num_samples = len(indices) * self.upsample_factor
            indices = weighted_dist.sample((num_samples,))
        elif self.split == 'train': # we only do weighting / upsampling on train, so this will make comparison easier
            indices = indices.repeat(self.upsample_factor)


        x = self.data.iloc[indices]['x'].values
        y = self.data.iloc[indices]['y'].values
        z = self.data.iloc[indices]['z'].values

        x = torch.tensor(list(x), dtype=torch.float32)
        y = torch.tensor(list(y), dtype=torch.float32).unsqueeze(-1)
        z = torch.tensor(list(z), dtype=torch.float32).unsqueeze(-1)
        
        if self.knowledge_type == "z":
            knowledge = z
        else: 
            knowledge = self.get_knowledge(idx)

        if self.use_optimal_rep: 
            x = x.sum(axis=-1).unsqueeze(-1)

        return x, y, knowledge, z

    def set_use_optimal_rep(self):
        self.use_optimal_rep = True
        self.dim_x = 1
        
    def add_weights(self, weights): 
        if self.weighted: 
            raise Exception("This dataset already has a set of weights.")

        self.weighted = True
        self.weights = weights
        self.upsample_factor = 10   
        # self.set_use_optimal_rep()

    def get_knowledge(self, task): 
        knowledge = self.data_generating_params[task]
        knowledge = knowledge.view(2, 1)
        indicator = torch.eye(2)
        knowledge = torch.cat([indicator, knowledge], dim=1)

        if self.knowledge_type == 'bc':
            revealed = np.random.choice([0, 1])
            mask = torch.zeros((2, 1))
            mask[revealed] = 1.0
            knowledge = knowledge * mask
        elif self.knowledge_type == 'b':
            knowledge = knowledge[0, :].unsqueeze(0)
        elif self.knowledge_type == 'c':
            knowledge = knowledge[1, :].unsqueeze(0)
        elif self.knowledge_type == 'full':
            pass
        elif self.knowledge_type  == 'none':
            knowledge = torch.zeros_like(knowledge)
        else:
            raise NotImplementedError

        return knowledge