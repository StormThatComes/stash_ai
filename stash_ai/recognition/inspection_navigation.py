"""Bounded session navigation without serializing database handles or credentials."""
class History:
    def __init__(self,limit=50):
        self.entries=[];self.index=-1;self.limit=limit

    def record(self,state):
        key=(state['scene'],state.get('filters'),state.get('selected'))
        if self.index>=0:
            old=self.entries[self.index]
            if key==(old['scene'],old.get('filters'),old.get('selected')):
                self.entries[self.index]=state
                return
        del self.entries[self.index+1:]
        self.entries.append(state)
        self.entries=self.entries[-self.limit:]
        self.index=len(self.entries)-1

    def move(self,direction):
        target=self.index+direction
        if not 0<=target<len(self.entries):return None
        self.index=target
        return self.entries[target]
