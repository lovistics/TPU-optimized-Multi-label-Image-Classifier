import warnings
import torch
warnings.filterwarnings("ignore")

def check_tpu_status():
    """Check TPU availability and print detailed information."""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("\nTPU Information:")
        print(f"TPU Device: {device}")
        print("TPU is available and configured correctly")
        return device
    except Exception as e:
        print("\nWARNING: No TPU detected! Error:", str(e))
        print("\nFalling back to GPU/CPU...")
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
