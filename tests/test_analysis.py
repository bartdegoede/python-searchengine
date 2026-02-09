from search.analysis import (
    analyze,
    lowercase_filter,
    punctuation_filter,
    stem_filter,
    stopword_filter,
    tokenize,
)


def test_tokenize():
    assert tokenize("hello world") == ["hello", "world"]
    assert tokenize("single") == ["single"]
    assert tokenize("") == []


def test_lowercase_filter():
    assert lowercase_filter(["Hello", "WORLD"]) == ["hello", "world"]
    assert lowercase_filter(["already"]) == ["already"]


def test_punctuation_filter():
    assert punctuation_filter(["hello!", "world."]) == ["hello", "world"]
    assert punctuation_filter(["it's"]) == ["its"]
    assert punctuation_filter(["clean"]) == ["clean"]


def test_stopword_filter():
    assert stopword_filter(["the", "quick", "fox"]) == ["quick", "fox"]
    assert stopword_filter(["the", "a", "in"]) == []
    assert stopword_filter(["python", "programming"]) == ["python", "programming"]


def test_stem_filter():
    result = stem_filter(["running", "cats", "programming"])
    assert result == ["run", "cat", "program"]


def test_analyze_full_pipeline():
    result = analyze("The quick Brown FOX jumped!")
    # "the" is a stopword, punctuation stripped, lowercased, stemmed
    assert "the" not in result
    assert "quick" in result
    assert "brown" in result
    assert "fox" in result
    assert "jump" in result


def test_analyze_filters_empty_tokens():
    # Punctuation-only tokens should be filtered out
    result = analyze("... --- !!!")
    assert result == []
