import pytest
import random


__all__ = ['train_complex', 'versions', 'addresses', 'versions_more', 'addresses_more', 'versions_slim']

@pytest.fixture
def train_complex():
    return [
        "0",
        "9",
        "",
        "123",
        "apple",
        "",
        "@",
        "中華文化",
        "   "
    ]

@pytest.fixture
def versions():
    with open('tests/data/version.txt', 'r') as f:
        whole_patterns = f.read().split('\n')
    return random.sample(whole_patterns, 20) # Tuned for fado+ai

@pytest.fixture
def addresses():
    with open('tests/data/address.txt', 'r') as f:
        whole_patterns = f.read().split('\n')
    return random.sample(whole_patterns, 10) # Tuned for fado+ai


@pytest.fixture
def versions_more():
    with open('tests/data/version.txt', 'r') as f:
        whole_patterns = f.read().split('\n')
    return random.sample(whole_patterns, 100) # Tuned for pure-ai

@pytest.fixture
def addresses_more():
    with open('tests/data/address.txt', 'r') as f:
        whole_patterns = f.read().split('\n')
    return random.sample(whole_patterns, 150) # Tuned for pure-ai


@pytest.fixture
def versions_slim():
    with open('tests/data/version.txt', 'r') as f:
        whole_patterns = f.read().split('\n')
    return random.sample(whole_patterns, 15) # Tuned for pure-ai