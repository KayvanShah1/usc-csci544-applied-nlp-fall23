import pytest


def pytest_addoption(parser):
    parser.addoption("--greedy", action="store", default="./out/greedy.json")
    parser.addoption("--viterbi", action="store", default="./out/viterbi.json")
    parser.addoption("--model", action="store", default="./out/hmm.json")
    parser.addoption("--vocab", action="store", default="./out/vocab.txt")
    parser.addoption("--dev", action="store", default="./data/dev.json")
    parser.addoption("--test", action="store", default="./data/test.json")


@pytest.fixture(scope="session")
def greedy(request):
    name_value = request.config.option.greedy
    if name_value is None:
        pytest.skip()
    return name_value


@pytest.fixture(scope="session")
def viterbi(request):
    name_value = request.config.option.viterbi
    if name_value is None:
        pytest.skip()
    return name_value


@pytest.fixture(scope="session")
def model(request):
    name_value = request.config.option.model
    if name_value is None:
        pytest.skip()
    return name_value


@pytest.fixture(scope="session")
def vocab(request):
    name_value = request.config.option.vocab
    if name_value is None:
        pytest.skip()
    return name_value


@pytest.fixture(scope="session")
def dev(request):
    name_value = request.config.option.dev
    if name_value is None:
        pytest.skip()
    return name_value


@pytest.fixture(scope="session")
def test(request):
    name_value = request.config.option.test
    if name_value is None:
        pytest.skip()
    return name_value
