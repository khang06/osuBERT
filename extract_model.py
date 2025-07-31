import torch
from model import LitOsuBert, LitOsuBertClassifier
def main():
    # Requires TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    model = LitOsuBertClassifier.load_from_checkpoint("final_v7_ai.ckpt").model
    model.save_pretrained("export")

if __name__ == "__main__":
    main()
