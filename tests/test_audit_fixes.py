import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.utils import (
    _assert_hydrated_checkpoint,
    format_prediction,
    preprocess_image,
)


def test_bgr_array_is_converted_to_rgb():
    # Pure red in OpenCV's BGR order is [0, 0, 255]; after conversion it must be
    # red in channel 0, matching the RGB PIL path the model was trained on.
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    tensor = preprocess_image(bgr)[0]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalised = tensor * std + mean

    assert unnormalised[0].mean().item() == pytest.approx(1.0, abs=1e-3)
    assert unnormalised[2].mean().item() == pytest.approx(0.0, abs=1e-3)


def test_pil_image_matches_array_path():
    pil = Image.new("RGB", (8, 8), (255, 0, 0))
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    torch.testing.assert_close(preprocess_image(pil), preprocess_image(bgr))


def test_format_prediction_picks_top_class():
    assert format_prediction({"With Mask": 0.91, "Without Mask": 0.09}) == (
        "With Mask: 91.00%"
    )
    assert format_prediction({"With Mask": 0.2, "Without Mask": 0.8}) == (
        "Without Mask: 80.00%"
    )


def test_missing_checkpoint_reports_path():
    with pytest.raises(FileNotFoundError, match="MASK_DETECTOR_MODEL_PATH"):
        _assert_hydrated_checkpoint("/nonexistent/model.pt")


def test_lfs_pointer_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        path.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:0a936a7f1e57fa329c0880fa3f54557fc015be624de8a9e7eb99823e77d79f01\n"
            b"size 20305172\n"
        )
        with pytest.raises(RuntimeError, match="git lfs pull"):
            _assert_hydrated_checkpoint(str(path))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from torchvision import models

    # Mirror the classifier built by scripts.utils.get_model so state_dict matches.
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(960, 1280),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(1280, 640),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(640, 2),
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setenv("MASK_DETECTOR_MODEL_PATH", str(checkpoint))

    from fastapi.testclient import TestClient

    import app.main as main

    with TestClient(main.app) as test_client:
        yield test_client


def _png_bytes(size=(32, 32)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buffer, "PNG")
    return buffer.getvalue()


def test_health_reports_loaded_model(client):
    payload = client.get("/health").json()
    assert payload == {"status": "ok", "model_loaded": True}


def test_predict_returns_class_probabilities(client):
    response = client.post(
        "/predict/", files={"file": ("a.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"With Mask", "Without Mask"}
    assert sum(body.values()) == pytest.approx(1.0, abs=1e-5)


def test_predict_rejects_disallowed_content_type(client):
    response = client.post(
        "/predict/", files={"file": ("a.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_rejects_undecodable_image(client):
    response = client.post(
        "/predict/", files={"file": ("a.png", b"not an image", "image/png")}
    )
    assert response.status_code == 400


def test_predict_rejects_oversized_upload(client, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024)
    response = client.post(
        "/predict/", files={"file": ("a.png", b"x" * 2048, "image/png")}
    )
    assert response.status_code == 413


def test_early_stopping_saves_only_improvements(tmp_path):
    import train as train_module

    save_path = tmp_path / "best.pt"
    stopper = train_module.EarlyStopping(patience=2, min_delta=0.001, save_path=str(save_path))
    model = torch.nn.Linear(2, 2)

    assert stopper(0.90, model) is True
    assert save_path.exists()
    first_write = save_path.stat().st_mtime_ns

    assert stopper(0.89, model) is False
    assert save_path.stat().st_mtime_ns == first_write

    assert stopper(0.88, model) is False
    assert stopper.early_stop is True


def test_seed_everything_is_deterministic():
    import train as train_module

    train_module.seed_everything(42)
    first = torch.rand(3)
    train_module.seed_everything(42)
    torch.testing.assert_close(first, torch.rand(3))
