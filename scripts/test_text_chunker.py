import pytest

from app.infrastructure.text_chunker import TextChunker


def test_empty_text_returns_empty_list():
    chunker = TextChunker()

    result = chunker.chunk(text="")

    assert result == []


def test_text_shorter_than_chunk_size_returns_one_chunk():
    chunker = TextChunker(
        chunk_size=100,
        overlap=20,
    )

    result = chunker.chunk(
        text="This is a short piece of text."
    )

    assert len(result) == 1
    assert result[0].text == "This is a short piece of text."
    assert result[0].chunk_index == 0


def test_text_is_split_into_chunks():
    chunker = TextChunker(
        chunk_size=10,
        overlap=2,
    )

    result = chunker.chunk(
        text="ABCDEFGHIJKLMNOPQRST"
    )

    assert [chunk.text for chunk in result] == [
        "ABCDEFGHIJ",
        "IJKLMNOPQR",
        "QRST",
    ]


def test_chunk_indexes_are_sequential():
    chunker = TextChunker(
        chunk_size=10,
        overlap=2,
    )

    result = chunker.chunk(
        text="ABCDEFGHIJKLMNOPQRST"
    )

    assert [
        chunk.chunk_index
        for chunk in result
    ] == [0, 1, 2]


def test_overlap_is_preserved():
    chunker = TextChunker(
        chunk_size=10,
        overlap=2,
    )

    result = chunker.chunk(
        text="ABCDEFGHIJKLMNOPQRST"
    )

    assert result[0].text[-2:] == result[1].text[:2]
    assert result[1].text[-2:] == result[2].text[:2]


def test_chunk_size_is_respected():
    chunker = TextChunker(
        chunk_size=10,
        overlap=2,
    )

    result = chunker.chunk(
        text="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    assert all(
        len(chunk.text) <= 10
        for chunk in result
    )


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=100,
            overlap=-1,
        )


def test_overlap_equal_to_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=100,
            overlap=100,
        )


def test_overlap_greater_than_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=100,
            overlap=101,
        )

def test_chunks_realistic_paper_text():
    chunker = TextChunker(
        chunk_size=100,
        overlap=20,
    )

    text = (
        "Retrieval augmented generation combines information "
        "retrieval with language generation. "
        "Scientific papers contain long-form technical information "
        "that must be divided into smaller pieces before embedding."
    )

    result = chunker.chunk(text=text)

    assert len(result) > 1

    assert all(
        chunk.text
        for chunk in result
    )

    assert [
        chunk.chunk_index
        for chunk in result
    ] == list(range(len(result)))