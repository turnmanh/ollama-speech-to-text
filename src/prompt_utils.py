from langchain_core.messages import HumanMessage, SystemMessage


def get_system_message(text: str = None) -> SystemMessage:
    """Provide a System message.

    Args:
        text: Content of the message.

    Returns:
        SystemMessage: System message to the model.
    """
    if text is None:
        text = """
            You are a helpful and friendly AI assistant. You are polite, 
            respectful, and aim to provide concise responses with as little 
            words as possible.
            """
    return SystemMessage(content=text)

def get_human_message(data: dict) -> HumanMessage:
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return HumanMessage(content=content_parts)

def get_prompt(data: dict) -> dict:
    """Provide a prompt for the model.

    Args:
        data: Data to be used in the prompt.

    Returns:
        dict: Prompt for the model.
    """
    human_message = get_human_message(data)
    system_message = get_system_message("Which castle is shown in the image?")

    return [system_message, human_message]
