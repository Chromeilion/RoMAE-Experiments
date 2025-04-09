import pandas as pd
import numpy as np
import torch
from ucimlrepo import fetch_ucirepo
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from roma.utils import gen_mask, prepare_positions

class CustomCollate:
    """
    Custom collate fn for use with the PLASTICC dataset.
    Recieves list of dictionaireees my dataset made
    """
    def __init__(self, t_jitter: float = 0., f_jitter: float = 0.):
        self.t_jitter = t_jitter
        self.f_jitter = f_jitter

    def __call__(self, batch):
        keys_of_interest = ["positions", "values","mask","label"]
        batch = [{key: i[key] for key in keys_of_interest} for i in batch]
        keys = list(batch[0].keys())
        labels = torch.stack([i["label"] for i in batch])
        keys.remove("label")
        pad_amount = max([i["values"].shape[0] for i in batch])
        p_batch = {'values': torch.stack([F.pad(i['values'].squeeze(), (0, pad_amount - i['values'].shape[0])) for i in batch])[..., None, None, None]}
        
        pad_amount = max([i["positions"].shape[1] for i in batch])
        p_batch['positions']= torch.stack([F.pad(i['positions'], (0, pad_amount - i['positions'].shape[1])) for i in batch])
        
        pad_amount = max([i["mask"].shape[0] for i in batch])
        p_batch['mask']= torch.stack([F.pad(i['mask'], (0, pad_amount - i['mask'].shape[0])) for i in batch])
        
        num_masks     = np.sum(p_batch['mask'].numpy(), axis=1)
        max_num_masks = np.max(num_masks)
        # Maximum number of masked values in the batch. number of masked values to add in in each sample
        tot_m         = abs(num_masks - max_num_masks)

        new_masks     = torch.zeros(size = (p_batch['positions'].shape[0], max_num_masks), dtype=torch.bool)

        for i in range(p_batch['positions'].shape[0]):
            new_masks[i, :tot_m[i]] = True
        p_batch['mask'] = torch.cat((p_batch['mask'], new_masks), dim=1)

        # print(p_batch['mask'].shape, p_batch['values'].shape, p_batch['positions'].shape)

        p_batch["values"] = torch.cat([p_batch["values"],torch.zeros(size = (p_batch['positions'].shape[0], max_num_masks,1,1,1))], dim=1)

        p_batch['positions'] = torch.cat((p_batch['positions'], torch.zeros(size = (p_batch['positions'].shape[0], 2,max_num_masks))), dim=2)



        pad_mask = torch.zeros_like(p_batch["mask"], dtype=torch.bool)
        


        for i, j in enumerate(batch):
            indices = torch.where(batch[i]["mask"] == 1)[0][-1]
            pad_mask[i, indices:] = True



            
        # torch.ones_like(new_masks, dtype=torch.bool)
        # print(pad_mask)

        # p_batch["label"] = labels
        # p_batch["pad_mask"] = pad_mask
        # p_batch["values"] = p_batch["values"][..., None, None, None]
        # if self.mask_ratio is not None:
        #     p_batch["mask"] = gen_mask(self.mask_ratio, p_batch["pad_mask"])
        # p_batch["positions"] = [p_batch["t"], p_batch["h"], None]
        # p_batch["positions"] = prepare_positions(len(batch), p_batch["positions"])
        # p_batch.pop("t", None), p_batch.pop("h", None)

        # p_batch["positions"] += torch.randn(p_batch["positions"].size())*self.t_jitter**0.5
        # p_batch["values"] += torch.randn(p_batch["values"].size())*self.f_jitter**0.5

        if torch.isnan(p_batch["values"]).any():

            print("In collate x contains NaNs:")
            print("x contains NaNs:", p_batch["values"])




        return p_batch


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ElectricityDataset(Dataset):
    """
    Dataset for processing the UCI household electricity data for interpolation experiments.
    
    For each day in the dataset, the following is computed:
      - "ground_truth": (num_pred, 7) tensor of dense values (sampled every ~16 minutes).
      - "times": (1, num_pred) tensor of corresponding normalized time stamps.
      - "values": tensor of combined irregular observations and regular predictions,
                  stored with extra singleton dimensions.
      - "mask": tensor indicating missing observations (0.0 for observed, 1.0 for missing) 
                for the combined observations.
      - "positions": (2, N) tensor; first row contains normalized time stamps and second row 
                     the feature indices for each observation.
      - "label": placeholder tensor.
    
    The processing pipeline includes:
      1. Converting to numeric and interpolating missing values.
      2. Min-max normalization of the 7 features:
         Global_active_power, Global_reactive_power, Voltage, Global_intensity,
         Sub_metering_1, Sub_metering_2, Sub_metering_3.
      3. Resampling the data to a 1–minute frequency and grouping by day.
      4. For each day:
         - Creating a full 1–minute time grid.
         - Sampling regular indices (every 16 minutes with a small random shift) to form the
           ground truth.
         - Simulating irregular sampling via exponential gaps (with mean gap λ).
         - At each irregular time, randomly selecting one feature to observe.
         - Combining the irregular and regular (predicted) observations.
    """
    def __init__(self, X, lam=20):
        # Ensure that the DataFrame index is a DatetimeIndex.
        if not isinstance(X.index, pd.DatetimeIndex):
            if 'Date' in X.columns and 'Time' in X.columns:
                X['datetime'] = pd.to_datetime(X['Date'] + ' ' + X['Time'], dayfirst=True)
                X.set_index('datetime', inplace=True)
        
        # Define the 7 relevant features.
        features = ["Global_active_power", "Global_reactive_power", "Voltage",
                    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
               # Replace NaNs or Infs in the dataset with zeros
        X = X.replace([np.nan, np.inf, -np.inf], 0)
        
        X = X[features]
        
        # Convert columns to numeric and interpolate missing values.
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Normalize each feature to [0, 1] using min-max normalization over the entire dataset.
        X_norm = (X - X.min()) / (X.max() - X.min())

        X_norm = X_norm.replace([np.nan, np.inf, -np.inf], 0)
        
        # Group the data by day.
        groups = X_norm.groupby(X_norm.index.date)
        self.samples = []
        

        for day, group in groups:
            # Create a complete 1–minute index for the day (1440 minutes).
            day_index = pd.date_range(start=pd.to_datetime(day), periods=1440, freq='T')
            group = group.reindex(day_index)
            group = group.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
            
            # Dense ground truth: shape (1440, 7)
            dense_values = group.values
            T = dense_values.shape[0]
            positions = np.linspace(0, 1, T)  # normalized time grid
            
            # Initialize mask (1.0 indicates observed; we later mark missing with 0.0)
            mask = np.ones_like(dense_values, dtype=np.float32)
            
            # Select regular indices for ground truth (every 16 minutes with a random shift)
            random_shift = np.random.randint(3)
            rnd_idx = np.arange(random_shift, T, 16)
            pred_values = dense_values[rnd_idx]  # shape: (num_pred, 7)
            
            # Build predicted positions:
            # For each prediction, record its normalized time (repeated for each of the 7 features)
            pred_times = positions[rnd_idx][:, None].repeat(7, axis=1).flatten()
            pred_features = np.tile(np.arange(7), rnd_idx.shape[0])
            pred_positions = np.vstack([pred_times, pred_features])
            
            # Simulate irregular sampling using exponential gaps.
            obs_indices = []
            t = 0
            while t < T:
                dt = int(np.round(np.random.exponential(scale=lam)))
                t += max(1, dt)
                if t < T and t not in rnd_idx:
                    obs_indices.append(t)
            
            # For each observation index, randomly choose one feature.
            obs_vals = []
            pos_list = []
            for idx in obs_indices:
                feat = np.random.choice(7)
                mask[idx, feat] = 0.0  # mark this feature as unobserved
                obs_vals.append(dense_values[idx, feat])
                pos_list.append((positions[idx], feat))
            
            if len(obs_vals) > 0:
                obs_vals = np.array(obs_vals)
                pos_list = np.array(pos_list).T  # shape: (2, number of irregular observations)
            else:
                obs_vals = np.array([])
                pos_list = np.empty((2, 0))
            
            # Combine irregular observations with the predicted (regular) observations.
            combined_vals = np.concatenate([obs_vals, pred_values.flatten()])
            combined_positions = np.concatenate([pos_list, pred_positions], axis=1)
            
            # Create a mask for the combined observations.
            # Here we set the last part (corresponding to pred_values) as True (observed)
            mask_combined = np.zeros_like(combined_vals, dtype=bool)
            mask_combined[-pred_positions.shape[1]:] = True
            mask_combined = mask_combined  
            
            sample_dict = {
                "ground_truth": torch.tensor(pred_values, dtype=torch.float32),
                "times": torch.tensor(positions[rnd_idx], dtype=torch.float32).unsqueeze(0),
                "values": torch.tensor(combined_vals, dtype=torch.float32)[..., None, None, None],
                "mask": torch.tensor(mask_combined, dtype=torch.bool),
                "positions": torch.tensor(combined_positions, dtype=torch.float32),
                "label": torch.tensor(0, dtype=torch.long),
            }
            self.samples.append(sample_dict)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]
