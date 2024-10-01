import base64
from io import BytesIO
from PIL import Image


def convert_to_base64(pil_image: Image) -> str:
    """
    Convert PIL images to Base64 encoded strings.

    Args:
        pil_image: Input image to decode.

    Returns:
        str: Image as base64 encoded string.
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prep_image(image_path: str) -> str:
    """Prepare image encoded as str."""
    pil_image = Image.open(image_path)
    image_b64 = convert_to_base64(pil_image)
    return image_b64
