# Import the regular expressions module
import re

# Import types from the typing module for type annotations
from typing import Pattern, Union, List, Tuple, Iterator, Optional, Match

# Define a class to encapsulate regular expression operations
class Regex():
    def __init__(self):
        # Constructor of the class, does nothing in this case
        pass
    
    # Compile a regular expression pattern into a Pattern object
    def compile(self, pattern: str) -> Pattern:
        return re.compile(pattern)
    
    # Search for the pattern in the text and return the first match found
    def search(self, pattern: Union[str, Pattern], text: str) -> Optional[Match]:
        return re.search(pattern, text)
    
    # Find all occurrences of the pattern in the text and return an iterator of matches
    def finditer(self, pattern: Union[str, Pattern], text: str) -> Iterator[Match]:
        return re.finditer(pattern, text)
    
    # Check if the entire text matches the pattern
    def fullmatch(self, pattern: Union[str, Pattern], text: str) -> Optional[Match]:
        return re.fullmatch(pattern, text)
    
    # Check if the start of the text matches the pattern
    def match(self, pattern: Union[str, Pattern], text: str) -> Optional[Match]:
        return re.match(pattern, text)
    
    # Find all occurrences of the pattern in the text and return them as a list of strings
    def findall(self, pattern: Union[str, Pattern], text: str) -> List[str]:
        return re.findall(pattern, text)
    
    # Replace all occurrences of the pattern in the text with the replacement string
    def sub(self, pattern: Union[str, Pattern], repl: str, text: str) -> str:
        return re.sub(pattern, repl, text)
    
    # Replace all occurrences of the pattern in the text with the replacement string and return a tuple with the new text and the number of substitutions made
    def subn(self, pattern: Union[str, Pattern], repl: str, text: str) -> Tuple[str, int]:
        return re.subn(pattern, repl, text)
    
    # Split the text into a list of strings using the pattern as the delimiter
    def split(self, pattern: Union[str, Pattern], text: str) -> List[str]:
        return re.split(pattern, text)

# Example usage of the Regex class
# if __name__ == "__main__":
#     regex = Regex()
#     pattern = r'\bword\b'

#     text = "This is a word in a sentence."
    
#     # Examples of using the methods of the Regex class
#     print("search:", regex.search(pattern, text))  # Output: <re.Match object; span=(10, 14), match='word'>, span is the start and end index of the match
#     print("findall:", regex.findall(pattern, text))  # Output: ['word']
#     print("sub:", regex.sub(pattern, 'replacement', text))  # Output: "This is a replacement in a sentence."
#     print("split:", regex.split(r'\s+', text))  # Output: ['This', 'is', 'a', 'word', 'in', 'a', 'sentence.']
