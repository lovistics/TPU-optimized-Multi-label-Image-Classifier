def create_optimized_dataloaders(
    train_img_dir: str,
    val_img_dir: str,
    train_annotations: str,
    val_annotations: str,
    batch_size: int,
    num_workers: int,
    cache_dir: str = None,
    preload_memory: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders with augmentation."""

    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # Larger resize for random crop
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CachedVOCDataset(
        image_dir=train_img_dir,
        annotation_file=train_annotations,
        transform=train_transform,
        cache_dir=cache_dir,
        preload_memory=preload_memory,
        num_workers=num_workers
    )

    val_dataset = CachedVOCDataset(
        image_dir=val_img_dir,
        annotation_file=val_annotations,
        transform=val_transform,
        cache_dir=cache_dir,
        preload_memory=preload_memory,
        num_workers=num_workers
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    return train_loader, val_loader