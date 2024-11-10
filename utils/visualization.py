def display_predictions(model, dataloader, device, num_images=6):
    """Display predictions for random images that contain trained labels."""
    model.eval()

    # Initialize lists to store candidates
    candidates = {
        'images': [],
        'labels': []
    }

    # Collect candidate images that have at least one of the trained labels
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            # Check if image has any of our trained labels
            if torch.any(label > 0):  # Contains at least one positive label
                candidates['images'].append(img)
                candidates['labels'].append(label)

        if len(candidates['images']) >= num_images * 3:  # Get enough candidates
            break

    # Convert candidates to tensors
    candidate_images = torch.stack(candidates['images'])
    candidate_labels = torch.stack(candidates['labels'])

    # Randomly select from candidates
    total_candidates = len(candidate_images)
    if total_candidates < num_images:
        print(f"Warning: Only found {total_candidates} images with relevant labels")
        num_images = total_candidates

    random_indices = torch.randperm(total_candidates)[:num_images]

    # Select random images and labels from candidates
    images = candidate_images[random_indices]
    labels = candidate_labels[random_indices]

    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        if str(device).startswith('xla'):
            import torch_xla.core.xla_model as xm
            xm.mark_step()
            outputs = outputs.cpu()
        else:
            outputs = outputs.cpu()

        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).float()

    # Create grid of images
    rows = (num_images + 2) // 3  # Calculate needed rows
    cols = min(3, num_images)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    denorm = transforms.Normalize(-mean/std, 1.0/std)

    for idx, (img, pred, label) in enumerate(zip(images, predictions, labels)):
        # Denormalize and convert to numpy
        img_display = denorm(img).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)

        # Display image
        axes[idx].imshow(img_display)

        # Create prediction text
        pred_classes = [SELECTED_CLASSES[i] for i, p in enumerate(pred) if p > 0.5]
        true_classes = [SELECTED_CLASSES[i] for i, l in enumerate(label) if l > 0.5]

        pred_text = "Pred: " + ", ".join(pred_classes) if pred_classes else "Pred: none"
        true_text = "True: " + ", ".join(true_classes) if true_classes else "True: none"

        axes[idx].set_title(f"{pred_text}\n{true_text}", fontsize=8)
        axes[idx].axis('off')

    # Turn off any unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, val_metrics):
    """Plot training history with improved visualization."""
    plt.figure(figsize=(15, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot validation F1 score
    plt.subplot(1, 2, 2)
    plt.plot(val_metrics['f1'], 'r-', label='Validation F1')
    plt.title('Validation F1 Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()  # Display plot in notebook
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), bbox_inches='tight', dpi=300)
    plt.close()