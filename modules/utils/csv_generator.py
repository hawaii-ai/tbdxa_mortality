# TODO: Potentially add license.
# pylint: disable=no-member
""" Generator to extract data from .csv file. """

import joblib
import imageio
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras import utils


# TODO: Errors noticed: not scaling grip strength, not incluiiding current age as out column
class CombinedDataGen(utils.Sequence):
    """ Generate batches of paired x:y numpy arrays from csv source. 
    
    Args:
        data_file: Path to csv file
        out_mode: One of {'combined', 'meta', 'image'}.
            Default is 'combined'.
            What part of the dataset to return.
        mode: One of {'train', 'valid', 'test'}.
            Default is 'test'.
            Determines whether scalers are fit to data or previously fit scalers are applied for scaling. 
        shuffle: Boolean.
            Default is True. Determines whether rows in csv files are shuffled or not.
        scaler_dir: Path to directory. 
            Default is './models/scalers/
            Where scaler models are located (for 'test' or 'valid' mode) or will be saved (during 'train' mode).
        batch_size: Int. 
            Default is 1. Whether datapoints are returned in batches (batch_size>1) or individually (default).
            If used in conjunction with tf.data.Dataset.from_generator, batch_size = 1 is recommended. 

    Raises:
        AssertionError: If invalid arguments are provided for 'mode' or 'out_mode'.
    """
    def __init__(self,
                 data_file,
                 out_mode='combined',
                 mode='test',
                 shuffle=True,
                 scaler_dir='./models/scalers/',
                 batch_size=1):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.scaler_dir = scaler_dir

        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        assert out_mode in ['combined', 'meta', 'image']
        self.out_mode = out_mode

        self.DATA_COLS = ['10yr_death', 'tbdxa_file', 'participant_id']
        if self.out_mode in ['combined', 'meta']:
            self.META_COLS = [
                'age',
                'height',
                'weight',
                'bmi',
                'sex',
                'ethnicity',
                'fasted_glucose',
                'glucose',
                'fasted_insulin',
                'insulin',
                'hemoglobin_a1c',
                # TODO: mention in docs, specific column to HABC dataset
                'recalib_il6',
                'il6',
                'short_wlk_speed',
                'med_wlk_speed',
                'long_wlk_speed',
                'adl_disability_stat',
                'stair_disability_stat',
                'walk_disability_stat',
                'recent_falls',
                'num_recent_falls',
                'grip_strength'
            ]
            self.DATA_COLS += self.META_COLS

        self.df = pd.read_csv(data_file)
        assert self._validate_df_cols()

        if self.out_mode in ['combined', 'meta']:
            self._scale_data()

        self.indices = self.df.index.to_numpy().copy()
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _get_meta(self, batch):
        """Helper function for __getitem__ to only return metadata.
        
        Args:
            batch: pd.DataFrame.
                Subset of full dataset for current batch.

        Returns: Tuple (x, y).       
        """
        x_meta = []
        y_bin = []

        for row_idx in batch.index:
            x_meta.append(batch.loc[row_idx, self.META_COLS])
            y_bin.append(batch.loc[row_idx, '10yr_death'])

        x_meta = np.squeeze(np.array(x_meta))
        y_bin = np.array(y_bin)

        return {'meta_0': x_meta}, y_bin

    def _get_image(self, batch):
        """Helper function for __getitem__ to only return images.
        
        Args:
            batch: pd.DataFrame.
                Subset of full dataset for current batch.

        Returns: Tuple (x, y).       
        """
        x_img = []
        y_bin = []

        for row_idx in batch.index:
            im_file = batch.loc[row_idx, 'tbdxa_file']
            img = np.array(imageio.imread(im_file))
            try:
                assert img.dtype == np.uint8
            except:
                raise TypeError(
                    'Make sure images are saved in a format that can be read as uint8.'
                )
            # only fetch last two channels where image channels are high-energy/low-energy/air
            img = img[:, :, :2]
            try:
                assert img.shape == (1914, 654, 2)
            except AssertionError:
                ValueError(
                    'Image shape should be 1914x654. To support other shapes: modify cropping functions in setup_dataset.py and then comment out this assert.'
                )

            x_img.append(img)
            y_bin.append(batch.loc[row_idx, '10yr_death'])

        x_img = np.squeeze(np.array(x_img))
        y_bin = np.array(y_bin)

        return {'img_0': x_img}, y_bin

    def _get_combined(self, batch):
        """Helper function for __getitem__ to return metadata and images.
        
        Args:
            batch: pd.DataFrame.
                Subset of full dataset for current batch.

        Returns: Tuple (x, y).       
        """
        x_img = []
        x_meta = []
        y_bin = []

        for row_idx in batch.index:
            x_meta.append(batch.loc[row_idx, self.META_COLS])
            y_bin.append(batch.loc[row_idx, '10yr_death'])

            im_file = batch.loc[row_idx, 'tbdxa_file']
            img = np.array(imageio.imread(im_file))
            # only fetch last two channels where image channels are high-energy/low-energy/air
            img = img[:, :, :2]
            x_img.append(img)

        x_img = np.squeeze(np.array(x_img))
        x_meta = np.squeeze(np.array(x_meta))
        y_bin = np.array(y_bin)

        return {'meta_0': x_meta, 'img_0': x_img}, y_bin

    def _validate_df_cols(self):
        """Ensure that all necessary columns are present in input .csv file.
        
        Args:
            batch: pd.DataFrame.
                Subset of full dataset for current batch.

        Returns: Boolean.
        """
        in_cols = set(list(self.df.columns))
        for c in self.DATA_COLS:
            try:
                assert c in in_cols
            except AssertionError:
                print(
                    f'Column {c} missing in input csv file.\nPlease make sure each of the following columns is present in your csv file (column names need to match exactly):'
                )
                print('\n'.join(self.DATA_COLS))
                return False
        return True

    def _scale_data(self):
        """Scales all input column values into range [0,1].
        
        Args:
            batch: pd.DataFrame.
                Subset of full dataset for current batch.

        Returns: Boolean.
        """
        self._scale('height')
        self._scale('weight')
        self._scale('bmi')
        self._scale('fasted_glucose')
        self._scale('glucose')
        self._scale('fasted_insulin')
        self._scale('insulin')
        self._scale('hemoglobin_a1c')
        self._scale('recalib_il6')
        self._scale('il6')
        self._scale('age')
        self._scale('short_wlk_speed')
        self._scale('med_wlk_speed')
        self._scale('long_wlk_speed')
        self._scale('grip_strength')

    def _scale(self, col_name):
        """Helper function to scale individual columns in self.df.
        
        Either fits new sklearn scaler (if self.mode == 'train') or loads previously fit scaler and applies it to the column. 
        Args:
            col_name: String.
                Name of column to be scaled in self.df.
        """
        ign_index = self.df.loc[:, col_name].isnull()
        if self.mode == 'train':
            sclr = preprocessing.MinMaxScaler(feature_range=(0, 1))
            sclr.fit(self.df.loc[~ign_index,
                                 col_name].to_numpy().reshape(-1, 1))
            joblib.dump(sclr,
                        f'{self.scaler_dir}/{col_name}_minmax_scaler.joblib')

        else:
            sclr = joblib.load(
                f'{self.scaler_dir}/{col_name}_minmax_scaler.joblib')

        self.df.loc[~ign_index, col_name] = sclr.transform(
            self.df.loc[~ign_index, col_name].to_numpy().reshape(-1, 1))

    def __len__(self):
        """Returns length of iterator.

        Returns: Int.
        """
        return int(np.floor(self.indices.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """Function to get specific batch based on index.
        
        Called in sequence from 0 to self.__len__ to iterate over whole dataset. 

        Args:
            index: Int. Index of batch to retrieve.
        Returns: Tuple. 
            First item in tuple, the "input" batch, is determined by self.out_mode, second item is always the corresponding batch of labels.
        """
        batch_idx = self.indices[index * self.batch_size:(index + 1) *
                                 self.batch_size]
        batch = self.df.copy().loc[batch_idx, :]

        if self.out_mode == 'combined':
            return self._get_combined(batch)

        if self.out_mode == 'image':
            return self._get_image(batch)

        if self.out_mode == 'meta':
            return self._get_meta(batch)