"""Minimal tests for checkpoint downloader."""

from unittest.mock import patch

from src.utils.checkpoint_downloader import get_checkpoint


@patch(
    "src.utils.checkpoint_downloader.hf_hub_download", return_value="/cache/test.ckpt"
)
@patch(
    "src.utils.checkpoint_downloader.calculate_sha256",
    return_value="b3808a0ce2d5425797df358de854716d38ec7e32c9183a5931ef8109029fd9e4",
)
def test_download_success(mock_sha, mock_download):
    """Test successful checkpoint download."""
    result = get_checkpoint("mp_20_vae")

    assert str(result) == "/cache/test.ckpt"
    assert mock_download.called
