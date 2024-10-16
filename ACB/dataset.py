import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

class LoadTurbTowers():
    def get_dataloaders(self, n_train, n_val, n_test, batch_size, data_path):
        # inputs are all the continuous data
        # outputs are the continuous data at the locations furthest from the bubble (10 through 20)
        # parameters come from the input file, visc, diff, amp, bubble shape
        inputs_ = torch.load(f'{data_path}continuous.pt')
        parameters_ = torch.load(f'{data_path}discrete.pt')

        # get the channels from the several measurement locations
        outputs = torch.zeros(10000, 2, 10, 181)
        indices = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        # indices = torch.tensor([6])
        outputs[:,0,] = torch.index_select(inputs_[:,0,:], dim=1, index=indices)
        outputs[:,1,] = torch.index_select(inputs_[:,1,:], dim=1, index=indices)
        
        # normalize to [0,1]
        outputs = outputs.view(outputs.shape[0], 20, inputs_.shape[-1])
        mins = torch.min(torch.min(outputs, 0)[0], 1)[0]
        maxs = torch.max(torch.max(outputs, 0)[0], 1)[0]
        
        # Reshape min and max tensors to have dimensions compatible with tensor
        mins = mins.unsqueeze(0).unsqueeze(-1)  # Add dimensions to match tensor shape
        maxs = maxs.unsqueeze(0).unsqueeze(-1)  # Add dimensions to match tensor shape
        
        # Normalize the tensor
        outputs = (outputs - mins) / (maxs - mins)
        
        p_min = torch.min(parameters_, 0)[0]
        p_max = torch.max(parameters_, 0)[0]
        for param in range(6):
            parameters_[:,param] = (parameters_[:,param] - p_min[param]) / (p_max[param] - p_min[param])

        # First 1000 are validation, 1000:2000 are test, 2000:10,000 are training
        val_in = outputs[:n_val]
        test_in = outputs[n_val:n_val+n_test]
        train_in = outputs[-n_train:]
        
        val_param = parameters_[:n_val]
        test_param = parameters_[n_val:n_val+n_test]
        train_param = parameters_[-n_train:]
        
        val_dataset = TensorDataset(val_param, val_in, val_in)
        test_dataset = TensorDataset(test_param, test_in, test_in)
        train_dataset = TensorDataset(train_param, train_in, train_in)
        
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        return train_loader, val_loader, test_loader, mins, maxs, p_min, p_max, outputs.shape[-1]
    
    def get_datasets(self, n_train, n_val, n_test, batch_size, data_path):
        # inputs are all the continuous data
        # outputs are the continuous data at the locations furthest from the bubble (10 through 20)
        # parameters come from the input file, visc, diff, amp, bubble shape
        inputs_ = torch.load(f'{data_path}continuous.pt')
        parameters_ = torch.load(f'{data_path}discrete.pt')

        # get the channels from the several measurement locations
        outputs = torch.zeros(10000, 2, 10, 181)
        indices = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        # indices = torch.tensor([6])
        outputs[:,0,] = torch.index_select(inputs_[:,0,:], dim=1, index=indices)
        outputs[:,1,] = torch.index_select(inputs_[:,1,:], dim=1, index=indices)
        
        # normalize to [0,1]
        outputs = outputs.view(outputs.shape[0], 20, inputs_.shape[-1])
        mins = torch.min(torch.min(outputs, 0)[0], 1)[0]
        maxs = torch.max(torch.max(outputs, 0)[0], 1)[0]
        # Reshape min and max tensors to have dimensions compatible with tensor
        mins = mins.unsqueeze(0).unsqueeze(-1)  # Add dimensions to match tensor shape
        maxs = maxs.unsqueeze(0).unsqueeze(-1)  # Add dimensions to match tensor shape
        # Normalize the tensor
        outputs = (outputs - mins) / (maxs - mins)
        
        p_min = torch.min(parameters_, 0)[0]
        p_max = torch.max(parameters_, 0)[0]
        for param in range(6):
            parameters_[:,param] = (parameters_[:,param] - p_min[param]) / (p_max[param] - p_min[param])
        
        val_in = outputs[:n_val]
        test_in = outputs[n_val:n_val+n_test]
        train_in = outputs[-n_train:]
        
        val_param = parameters_[:n_val]
        test_param = parameters_[n_val:n_val+n_test]
        train_param = parameters_[-n_train:]
        
        
        return val_in, test_in, train_in, val_param, test_param, train_param, mins, maxs, p_min, p_max, outputs.shape[-1]
