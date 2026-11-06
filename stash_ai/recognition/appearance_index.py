"""Exact Hamming-radius lookup preserving the first inserted representative."""
class AppearanceIndex:
    def __init__(self):self.nodes=[]
    def add(self,value,payload):
        ordinal=len(self.nodes)
        self.nodes.append((value,payload,{}))
        if not ordinal:return
        current=0
        while True:
            other,_,children=self.nodes[current]
            distance=(value^other).bit_count()
            if distance not in children:children[distance]=ordinal;return
            current=children[distance]
    def first(self,value,distance):
        if not self.nodes or distance<=0:return None
        pending=[0];best=len(self.nodes)
        while pending:
            index=pending.pop()
            if index>=best:continue
            other,_,children=self.nodes[index]
            delta=(value^other).bit_count()
            if delta<distance:best=index
            # Strict radius and triangle inequality; no approximate neighbours.
            pending.extend(child for edge,child in children.items()
                if child<best and delta-distance<edge<delta+distance)
        return self.nodes[best][1] if best<len(self.nodes) else None
