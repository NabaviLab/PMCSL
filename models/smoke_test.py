import torch
from models import ProstateSemanticMIL


def main():
    model = ProstateSemanticMIL(
        visual_input_dim=2048,
        text_input_dim=768,
        hidden_dim=128,
        top_m=64,
    )
    features = {
        "5x": torch.randn(2, 180, 2048),
        "10x": torch.randn(2, 220, 2048),
        "20x": torch.randn(2, 260, 2048),
    }
    text_tokens = torch.randn(4, 32, 768)
    text_mask = torch.ones(4, 32, dtype=torch.long)
    out = model(features, text_tokens, text_mask)
    print("logits:", out["logits"].shape)
    print("slide embedding:", out["slide_embedding"].shape)
    print("grade prototypes:", out["grade_prototypes"].shape)
    assert out["logits"].shape == (2, 4)
    assert out["slide_embedding"].shape == (2, 128)
    assert out["grade_prototypes"].shape == (4, 128)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
