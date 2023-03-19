# Imports
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm

# Globals
ROOT = Path(r'C:\Users\Carlos\Downloads\camels_cl_4532001')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # This line checks if GPU is available

def loadDf(dataset):
    df=pd.read_csv(os.path.join(ROOT,dataset),index_col=0,parse_dates=True)
    return df
    
def load_forcing(basin: str) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data of a specific basin.

    :param basin: 8-digit code of basin as string.
    
    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin in m2 as integer.
    """
    
    # read-in data and convert date to datetime index
    PET=loadDf('pet_hargreaves_day.csv').iloc[:,-1]
    PP=loadDf('precip_cr2met_day.csv').iloc[:,-1]
    Tmax=loadDf('tmax_cr2met_day.csv').iloc[:,-1]
    Tmin=loadDf('tmin_cr2met_day.csv').iloc[:,-1]
    idx=Tmin.index
    idx=idx[idx>='1979-01-02']
    swe=loadDf('SWE_basin_79_23.csv').loc[idx]
    sca=loadDf('SCA_basin_79_23.csv').loc[idx]
    albedo=loadDf('SAlbedo_basin_79_23.csv').loc[idx]
    pres=loadDf('SfPress_basin_79_23.csv').loc[idx]
    df=pd.concat([PET,PP,Tmax,Tmin,swe,sca,albedo,pres],axis=1)
    df.columns=['pet(mm/d)','pp(mm/d)','tmax(C)','tmin(C)','swe','sca',
                'alb','press']

    # load area from header in m2
    dfArea=pd.read_csv(os.path.join(ROOT,'catchment_attributes.csv'),index_col=0)
    area=int(float(dfArea.loc['area_km2'][0])*1e6)
    return df, area


def load_discharge(basin: str, area: int) ->  pd.Series:
    """Load the discharge time series for a specific basin.

    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters
    
    :return: A pd.Series containng the catchment normalized discharge.
    """
        
    # get path of streamflow file file
    q=loadDf('q_m3s_day.csv').iloc[:,-1]
    q.columns=['QObs(mm/d)']
    q=q.loc[q.index>=pd.to_datetime('1979-01-01')]

    # normalize discharge from cubic feed per second to mm per day
    df = q * 86400 * 1e3 / area

    return df

@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new

class CamelsTXT(Dataset):
    """Torch Dataset for basic use of data from the CAMELS data set.

    This data set provides meteorological observations and discharge of a given
    basin from the CAMELS data set.
    """

    def __init__(self, basin: str, seq_length: int=365,period: str=None,
                 dates: List=None, means: pd.Series=None, stds: pd.Series=None):
        """Initialize Dataset containing the data of a single basin.

        :param basin: 8-digit code of basin as string.
        :param seq_length: (optional) Length of the time window of
            meteorological input provided for one time step of prediction.
        :param period: (optional) One of ['train', 'eval']. None loads the 
            entire time series.
        :param dates: (optional) List of pd.DateTimes of the start and end date 
            of the discharge period that is used.
        :param means: (optional) Means of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_means() on the data set.
        :param stds: (optional) Stds of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_stds() on the data set.
        """
        self.basin = basin
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds

        # load data into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output data from text files."""
        df, area = load_forcing(self.basin)
        dfQ=load_discharge(self.basin, area)
        df['QObs(mm/d)'] = dfQ.loc[df.index].values
        print(type(df.index[0]))
        
        if self.dates is not None:
            # If meteorological observations exist before start date
            # use these as well. Similiar to hydrological warmup period.
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            df = df[start_date:self.dates[1]]

        # if training period store means and stds
        if self.period == 'train':
            self.means = df.mean()
            self.stds = df.std()

        # extract input and output features from DataFrame
        x = np.array([df['pet(mm/d)'].values,
                      df['pp(mm/d)'].values,
                      df['tmax(C)'].values,
                      df['tmin(C)'].values,
                      df['swe'].values,
                      df['sca'].values,
                      df['alb'].values,
                      df['press'].values]).T
        y = np.array([df['QObs(mm/d)'].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self._local_normalization(x, variable='inputs')
        x, y = reshape_data(x, y, self.seq_length)

        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)
            
            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)
            
            # normalize discharge
            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['pet(mm/d)'],
                              self.means['pp(mm/d)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['swe'],
                              self.means['sca'],
                              self.means['alb'],
                              self.means['press']])
            stds = np.array([self.stds['pet(mm/d)'],
                             self.stds['pp(mm/d)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['swe'],
                             self.stds['sca'],
                             self.stds['alb'],
                             self.stds['press']])
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["QObs(mm/d)"]) /
                       self.stds["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['pet(mm/d)'],
                              self.means['pp(mm/d)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['swe'],
                              self.means['sca'],
                              self.means['alb'],
                              self.means('press')])
            stds = np.array([self.stds['pet(mm/d)'],
                             self.stds['pp(mm/d)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['swe'],
                             self.stds['sca'],
                             self.stds['alb'].
                             self.stds['press']])
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["QObs(mm/d)"] +
                       self.means["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

class Model(nn.Module):
    """Implementation of a single layer LSTM network"""
    
    def __init__(self, hidden_size: int, dropout_rate: float=0.0):
        """Initialize model
        
        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # create required layer
        self.lstm = nn.LSTM(input_size=8, hidden_size=self.hidden_size, 
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.
        
        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.
        
        :return: Tensor containing the network predictions
        """
        output, (h_n, c_n) = self.lstm(x)
        
        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1,:,:]))
        return pred

def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.tqdm_notebook(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        
def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.
    
    :return: Two torch Tensors, containing the observations and 
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)
            
    return torch.cat(obs), torch.cat(preds)
        
def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val

def main():
    #%%
    basin = '4532001' # can be changed to any 8-digit basin id contained in the CAMELS data set
    hidden_size = 10 # Number of LSTM cells
    dropout_rate = 0.0 # Dropout rate of the final fully connected Layer [0.0, 1.0]
    learning_rate = 1e-3 # Learning rate used to update the weights
    sequence_length = 365 # Length of the meteorological record provided to the network

    ##############
    # Data set up#
    ##############

    # Training data
    start_date = pd.to_datetime("1979-01-02", format="%Y-%m-%d")
    end_date = pd.to_datetime("2020-04-30", format="%Y-%m-%d")
    ds_train = CamelsTXT(basin, seq_length=sequence_length, period="train", dates=[start_date, end_date])
    tr_loader = DataLoader(ds_train, batch_size=256, shuffle=True)

    # Validation data. We use the feature means/stds of the training period for normalization
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    start_date = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2020-04-30", format="%Y-%m-%d")
    ds_val = CamelsTXT(basin, seq_length=sequence_length, period="eval", dates=[start_date, end_date],
                        means=means, stds=stds)
    val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False)

    # Test data. We use the feature means/stds of the training period for normalization
    start_date = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2020-04-30", format="%Y-%m-%d")
    ds_test = CamelsTXT(basin, seq_length=sequence_length, period="eval", dates=[start_date, end_date],
                        means=means, stds=stds)
    test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)

    #########################
    # Model, Optimizer, Loss#
    #########################

    # Here we create our model, feel free 
    model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    
    n_epochs = 67 # Number of training epochs

    for i in range(n_epochs):
        train_epoch(model, optimizer, tr_loader, loss_func, i+1)
        obs, preds = eval_model(model, val_loader)
        preds = ds_val.local_rescale(preds.cpu().numpy(), variable='output')
        nse = calc_nse(obs.numpy(), preds)
        tqdm.tqdm.write(f"Validation NSE: {nse:.2f}")
        
    # Evaluate on test set
    obs, preds = eval_model(model, test_loader)
    preds = ds_val.local_rescale(preds.cpu().numpy(), variable='output')
    obs = obs.numpy()
    nse = calc_nse(obs, preds)

    # Plot results
    start_date = ds_test.dates[0]
    end_date = ds_test.dates[1] + pd.DateOffset(days=1)
    date_range = pd.date_range(start_date, end_date)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(date_range,obs, label="observado")
    ax.plot(date_range,preds, label="modelado")
    ax.legend()
    ax.set_title(f"Basin {basin} - Test set NSE: {nse:.3f}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Fecha")
    _ = ax.set_ylabel("Caucal específico (mm/d)")
    #%%
    torch.save(model.state_dict(), os.path.join('..','modelos',
                                                'Combarbala.pth'))

def loadModel():
    modelo = Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)        
    modelo.load_state_dict(torch.load(os.path.join('.','model.pth')))
    return modelo