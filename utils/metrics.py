def evaluate_model(model, dataloader, device, plot_results=False):
    """Evaluate the model with multiple metrics and display results."""
    model.eval()
    all_preds = []
    all_labels = []
    is_tpu = str(device).startswith('xla')

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            if is_tpu:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
                outputs = outputs.cpu()
            else:
                outputs = outputs.cpu()

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).float()
            all_preds.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Print per-class metrics
    print("\nPer-class Metrics:")
    print("-" * 50)
    print(f"{'Class':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    if plot_results:
        plt.figure(figsize=(20, 4))

    for i, class_name in enumerate(SELECTED_CLASSES):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        prec = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        rec = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)

        # Print metrics for each class
        print(f"{class_name:<10} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")

        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)

        if plot_results:
            plt.subplot(1, 5, i + 1)
            metric_values = [acc, prec, rec, f1]
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
            plt.bar(metric_names, metric_values)
            plt.title(f'{class_name}')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)

    if plot_results:
        plt.tight_layout()
        plt.show()  # Display plot in notebook
        plt.savefig(os.path.join(OUTPUT_DIR, 'final_metrics.png'), bbox_inches='tight', dpi=300)
        plt.close()

    print("-" * 50)
    print(f"Average Metrics:")
    print(f"Accuracy:  {np.mean(metrics['accuracy']):.3f}")
    print(f"Precision: {np.mean(metrics['precision']):.3f}")
    print(f"Recall:    {np.mean(metrics['recall']):.3f}")
    print(f"F1-Score:  {np.mean(metrics['f1']):.3f}")

    return np.mean(metrics['f1'])