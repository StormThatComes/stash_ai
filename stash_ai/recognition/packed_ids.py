"""Packed canonical IDs and rank-indexed fields for a sealed observation set."""
from array import array
import uuid


def key(value):
    if isinstance(value,str) and len(value)==36 and value==value.lower():
        try:
            if tuple(value[i] for i in (8,13,18,23))==('-','-','-','-'):
                raw=bytes.fromhex(value.replace('-',''))
                if len(raw)==16:return b'\0'+raw
        except ValueError:pass
    return b'\1'+value.encode('utf8')


class Ranks(dict):
    def __getitem__(self,value):return super().__getitem__(key(value))
    def __setitem__(self,value,rank):super().__setitem__(key(value),rank)
    def __contains__(self,value):return super().__contains__(key(value))


class Order:
    def __init__(self):self.data=bytearray();self.other={}
    def __len__(self):return len(self.data)//16
    def append(self,value):
        raw=key(value)
        if raw[0]:self.other[len(self)]=value;self.data.extend(bytes(16))
        else:self.data.extend(raw[1:])
    def __getitem__(self,rank):
        if rank in self.other:return self.other[rank]
        return str(uuid.UUID(bytes=bytes(self.data[rank*16:(rank+1)*16])))


class Flags:
    def __init__(self,ranks,total):self.ranks=ranks;self.bits=bytearray(total)
    def add(self,value):self.bits[self.ranks[value]]=1
    def __contains__(self,value):
        return value in self.ranks and bool(self.bits[self.ranks[value]])


class Values:
    def __init__(self,ranks,total):
        self.ranks=ranks;self.ids=array('I',[0])*total;self.values=[None];self.lookup={}
    def __setitem__(self,value,item):
        index=self.lookup.get(item)
        if index is None:
            index=len(self.values);self.lookup[item]=index;self.values.append(item)
        self.ids[self.ranks[value]]=index
    def __getitem__(self,value):return self.values[self.ids[self.ranks[value]]]
    def get(self,value,default=None):
        if value not in self.ranks:return default
        result=self[value]
        return default if result is None else result
    def pop(self,value,default=None):
        if value not in self.ranks:return default
        result=self.get(value,default);self.ids[self.ranks[value]]=0;return result
