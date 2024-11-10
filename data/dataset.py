class CachedVOCDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform=None,
        cache_dir: str = None,
        preload_memory: bool = False,
        num_workers: int = 4
    ):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.preload_memory = preload_memory
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(image_dir), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load annotations
        self.data = pd.read_csv(annotation_file)
        self._verify_and_clean_data(num_workers)

        # Initialize caches
        self.image_cache: Dict[str, torch.Tensor] = {}
        self.cache_file = os.path.join(self.cache_dir, 'image_cache.pkl')

        if preload_memory:
            self._preload_images(num_workers)

    def _verify_and_clean_data(self, num_workers: int) -> None:
        """Verify images exist and are valid in parallel."""
        def verify_image(row) -> Tuple[int, bool]:
            img_path = os.path.join(self.image_dir, row['image_name'])
            try:
                with Image.open(img_path) as img:
                    img.verify()
                return row.name, True
            except:
                return row.name, False

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(verify_image, [row for _, row in self.data.iterrows()]))

        valid_indices = [idx for idx, valid in results if valid]
        self.data = self.data.loc[valid_indices].reset_index(drop=True)

    def _preload_images(self, num_workers: int) -> None:
        if os.path.exists(self.cache_file):
            print(f"Loading cached images from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.image_cache = pickle.load(f)
            return

        print("Preloading images into memory cache...")
        def load_image(img_name: str) -> Tuple[str, torch.Tensor]:
            img_path = os.path.join(self.image_dir, img_name)
            try:
                with Image.open(img_path).convert('RGB') as img:
                    if self.transform:
                        img_tensor = self.transform(img)
                    else:
                        img_tensor = transforms.ToTensor()(img)
                return img_name, img_tensor
            except Exception as e:
                warnings.warn(f"Failed to load {img_path}: {str(e)}")
                return img_name, None

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(load_image, self.data['image_name']),
                total=len(self.data),
                desc="Preloading images"
            ))

        self.image_cache = {name: tensor for name, tensor in results if tensor is not None}

        print(f"Saving image cache to {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.image_cache, f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.data.iloc[idx]['image_name']

        if self.preload_memory and img_name in self.image_cache:
            image = self.image_cache[img_name]
        else:
            try:
                img_path = os.path.join(self.image_dir, img_name)
                with Image.open(img_path).convert('RGB') as img:
                    image = self.transform(img) if self.transform else transforms.ToTensor()(img)
            except Exception as e:
                warnings.warn(f"Error loading image {img_path}: {str(e)}")
                image = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

        labels = torch.tensor([
            self.data.iloc[idx][class_name] for class_name in SELECTED_CLASSES
        ], dtype=torch.float32)

        return image, labels