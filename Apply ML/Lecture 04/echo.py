
"""
Explain the __init__ main lines

from: https://realpython.com/if-name-main-python/
"""

def echo(text: str, repetitions: int = 5) -> str:
    """Imitate a real-world echo."""
    echoed_text = ""
    for i in range(repetitions, 0, -1):
        echoed_text += f"{text[-i:]}\n"
    return f"{echoed_text.lower()}."


if __name__ == "__main__":
    text = 'hello'
    print(echo(text))

# text = 'hello'
# print(echo(text))