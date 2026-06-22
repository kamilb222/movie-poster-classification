"""
Custom PyTorch Dataset for Movie Poster Multi-Label Classification.
Loads images from local ./images/ folder and provides multi-hot encoded genre labels.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Paths
CSV_PATH = "data/MovieGenre_clean.csv"
IMAGES_DIR = "images"


def get_all_classes(df):
    """
    Extract sorted list of all unique genres from the dataset.
    """
    all_genres = set()
    for genres_str in df["Genre"].dropna():
        genres = [g.strip() for g in genres_str.split("|")]
        all_genres.update(genres)
    return sorted(list(all_genres))


def encode_labels(genres_str, all_classes):
    """
    Convert pipe-separated genre string to multi-hot float tensor.
    """
    label = torch.zeros(len(all_classes), dtype=torch.float32)
    
    if pd.isna(genres_str):
        return label
    
    genres = [g.strip() for g in genres_str.split("|")]
    for genre in genres:
        if genre in all_classes:
            idx = all_classes.index(genre)
            label[idx] = 1.0
    
    return label


class MoviePosterDataset(Dataset):
    """
    PyTorch Dataset for movie poster images with multi-label genre classification.
    """
    
    def __init__(self, df, all_classes, images_dir=IMAGES_DIR, transform=None):
        """
        Args:
            df: DataFrame with imdbId and Genre columns
            all_classes: Sorted list of all genre classes
            images_dir: Directory containing the poster images
            transform: Optional torchvision transforms
        """
        self.df = df.reset_index(drop=True)
        self.all_classes = all_classes
        self.images_dir = images_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imdb_id = row["imdbId"]
        genres_str = row["Genre"]
        
        # Load image
        image_path = os.path.join(self.images_dir, f"{imdb_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Encode labels
        label = encode_labels(genres_str, self.all_classes)
        
        return image, label


def get_transforms():
    """
    Get image transforms for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return train_transform, val_transform


def get_dataloaders(batch_size=32, val_split=0.2, num_workers=4):
    """
    Create train and validation DataLoaders.
    
    Args:
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader, all_classes
    """
    # Load cleaned CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")
    
    # Get all classes
    all_classes = get_all_classes(df)
    print(f"Number of classes: {len(all_classes)}")
    print(f"Classes: {all_classes}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create full dataset with train transforms (we'll handle val transforms separately)
    full_dataset = MoviePosterDataset(df, all_classes, transform=train_transform)
    
    # Split into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override transform for validation set (hacky but works)
    # Note: In production, you'd want a cleaner solution
    val_dataset.dataset = MoviePosterDataset(df, all_classes, transform=val_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, all_classes


if __name__ == "__main__":
    # Test the dataset
    train_loader, val_loader, all_classes = get_dataloaders(batch_size=4, num_workers=0)
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label example: {labels[0]}")
