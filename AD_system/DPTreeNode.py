'''
This is the tree node class for the inquiry system (for dynamic programming)
'''

class DPTreeNode:
    def __init__(self, answer, next_question):
        self.answer = answer
        self.next_question = next_question
        self.children = []
        self.parent = None
        self.entropy = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
    
    def look_up_children(self, answer):
        for child in self.children:
            if answer == child.answer:
                return child
        return None