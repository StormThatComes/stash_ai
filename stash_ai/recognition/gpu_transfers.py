"""Bounded owner-thread transfers; CPU decoding waits for explicit CUDA events."""
import threading


def to_device(tensor,device):
    if str(device).split(':')[0]=='cuda' and tensor.device.type=='cpu':
        tensor=tensor if tensor.is_pinned() else tensor.pin_memory()
        return tensor.to(device,non_blocking=True)
    return tensor.to(device)


class Readback:
    def __init__(self,cpu,event=None,source=None):
        self.cpu=cpu;self.event=event;self.source=source;self.lock=threading.Lock()
    def wait(self):
        with self.lock:
            if self.event is not None:self.event.synchronize();self.event=None
            self.source=None
            return self.cpu


class Transfers:
    def __init__(self,device):self.device=device;self.stream=None;self.pending=[]
    def read(self,tensor):
        if tensor.device.type!='cuda':return Readback(tensor.detach().cpu())
        import torch
        # Only one previous readback may overlap the current batch. Completed
        # CPU buffers belong to the decode future, not this transfer owner.
        while len(self.pending)>=1:self.pending.pop(0).wait()
        compute=torch.cuda.current_stream(self.device)
        if self.stream is None:self.stream=torch.cuda.Stream(device=self.device)
        cpu=torch.empty_like(tensor,device='cpu',pin_memory=True)
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(compute)
            cpu.copy_(tensor.detach(),non_blocking=True)
            tensor.record_stream(self.stream)
            event=torch.cuda.Event();event.record(self.stream)
        result=Readback(cpu,event,tensor);self.pending.append(result)
        return result
    def drain(self):
        try:
            for transfer in self.pending:transfer.wait()
        finally:self.pending.clear()
    def close(self):
        try:self.drain()
        finally:self.stream=None
