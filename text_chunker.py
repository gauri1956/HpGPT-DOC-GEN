def chunk_text(text, max_chars=3000, overlap=300):

    if len(text) <= max_chars:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = start + max_chars

        chunk = text[start:end]

        chunks.append(chunk)

        start += max_chars - overlap

    return chunks