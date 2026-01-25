import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

_DATASET_ROOT = Path(__file__).resolve().parent / 'datasets' / 'vimeo90k' / 'vimeo_septuplet'


class Video90k(Dataset):

    def __init__(
            self, 
            dataset_mode:str='train', 
            target_selection_mode:str='last', 
            num_frames:int=7, 
            num_channels:int=3, 
            img_resolution:(int)=(256, 448),
            downscale_factor:float=2,
            downscale_technique:str='linear'
            ):
        
        super(Video90k).__init__()

        self.dataset_mode = dataset_mode
        self.target_selection_mode = target_selection_mode
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.img_resolution = img_resolution
        self.downscale_factor = downscale_factor
        self.downscale_technique = downscale_technique
        self.dataset_path = _DATASET_ROOT / 'sequences'
        self.sequences = self.load_sequences()
        
    def __len__(self):
        return len(self.sequences)
    
    def load_sequences(self):
        train_seq = _DATASET_ROOT / 'sep_trainlist.txt'
        test_seq = _DATASET_ROOT / 'sep_testlist.txt'
        if self.dataset_mode == 'train':
            with open(train_seq, 'r') as train_seq_file:
                # read sequence lines, remove newline characters
                loaded_sequence = train_seq_file.readlines()
                loaded_sequence = [seq.strip() for seq in loaded_sequence]
            return loaded_sequence
        else:
            with open(test_seq, 'r') as test_seq_file:
                loaded_sequence = test_seq_file.readlines()
                loaded_sequence = [seq.strip() for seq in loaded_sequence]   

            return loaded_sequence
    
    def load_frames(self, sequence_path:Path):
        # hard-coded, bad practice but for now it's fine, we are SHUUURE (🇫🇷) to have 7 frames. 
        # i could have used os.listdir, but it's slower.

        h, w = self.img_resolution
        if self.downscale_technique == 'linear':
            interpolation = cv2.INTER_LINEAR
        elif self.downscale_technique == 'cubic':
            interpolation = cv2.INTER_CUBIC

        frames_high_res = np.zeros((self.num_frames, h, w, self.num_channels))
        frames_low_res = np.zeros((self.num_frames, h//self.downscale_factor, w//self.downscale_factor, self.num_channels))

        for img_idx in range(0, 7):
            img = cv2.cvtColor(cv2.imread(self.dataset_path/sequence_path/f'im{img_idx+1}.png'), cv2.COLOR_BGR2RGB) # h,w,c
            frames_high_res[img_idx] = img
            frames_low_res[img_idx] = cv2.resize(img, (w//self.downscale_factor, h//self.downscale_factor), interpolation=interpolation) # damn opencv, confusing to have w,h,c
        return frames_low_res, frames_high_res
    
    def select_target(self, frames_low_res:np.ndarray, frames_high_res:np.ndarray):
        if self.target_selection_mode == 'last':
            # For baseline: predict frame 7
            context_low_res = frames_low_res[:6]   # LR frames 1-6 (temporal context)
            target_low_res = frames_low_res[6]     # LR frame 7 (what to upscale). used in diffusion for conditioning
            target_high_res = frames_high_res[6]     # HR frame 7 (ground truth), the target to predict
            return context_low_res, target_low_res, target_high_res 

    def __getitem__(self, index):
        # load frames in high res (leave them as they are)
        frames_low_res, frames_high_res = self.load_frames(self.sequences[index]) # each frame has shape [7, H, W, C]
        
        # downsample to create the training sequence

        context_frames_low_res, target_low_res, target_high_res = self.select_target(frames_low_res, frames_high_res)

        context_frames_low_res = torch.from_numpy(np.stack(context_frames_low_res)) / 255.0
        target_low_res = torch.from_numpy(target_low_res) / 255.0
        target_high_res = torch.from_numpy(target_high_res) / 255.0

        return context_frames_low_res, target_low_res, target_high_res
