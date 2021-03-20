import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from vissl.data.data_helper import get_mean_image
from torch.utils.data import Dataset
import logging


class MyNewSourceDataset(Dataset):
    base_folder = 'CUB_200_2011/images'  # Base dataset path
    """
    add documentation on how this dataset works

    Args:
        add docstrings for the parameters
    """

    def __init__(self, cfg, data_source, path, split, dataset_name):
        super(MyNewSourceDataset, self).__init__()
        assert data_source in [
            "disk_filelist",
            "disk_folder",
            "my_data_source"
        ], "data_source must be either disk_filelist or disk_folder or my_data_source"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self.root = path
        # implement anything that data source init should do
        self._num_samples = 0  # set the length of the dataset
        self.loader = default_loader
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        self._num_samples = len(self.data)

    def __sample_data_train(self, group):
        return pd.DataFrame(group.sample(n=int(len(group))))

    def __sample_data_test(self, group):
        return pd.DataFrame(group.sample(n=int(len(group))))

    def _load_metadata(self):
        """
        Load the metadata (image list, class list and train/test split) of the CUB_200_2011 dataset
        """
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        if self.split == "train":
            self.data = self.data[self.data.is_training_img == 1]
            self.data = self.data.groupby('target').apply(self.__sample_data_train)
        else:
            self.data = self.data[self.data.is_training_img == 0]
            self.data = self.data.groupby('target').apply(self.__sample_data_test)

    def _check_integrity(self):
        """
        Verify if the dataset is present and loads accurately
        """
        try:
            self._load_metadata()
        except Exception:
            return False
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """
        try:
            sample = self.data.iloc[idx]
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            target = sample.target - 1  # Targets start at 1 by default, so shift to 0
            img = self.loader(path)  # Call the loader function to load the image
            is_success = True
        except Exception as e:
            logging.warning(
                f"Couldn't load: {self.image_dataset[idx]}. Exception: \n{e}"
            )
            is_success = False
            img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)

        # is_success should be True or False indicating whether loading data was successful or failed
        # loaded data should be Image.Image if image data
        return img, is_success
