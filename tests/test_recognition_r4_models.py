from types import SimpleNamespace
from stash_ai.recognition.resident_models import Stage,scope


def test_stage_loads_once_and_never_loads_during_cleanup():
    loaded=[]
    class Mock:
        def predict(self):return 12
        def close(self):loaded.append('closed')
    def factory():loaded.append('open');return Mock()
    config=SimpleNamespace(data={'execution':{'gpu_model_policy':'sequential'}})
    with scope(config):
        s=Stage('detector',factory,'cpu')
        s.close();assert not loaded
        assert s.predict()==s.predict()==12
        s.close()
        assert loaded==['open','closed']


def test_parked_weights_respect_cap_and_restore_once():
    from stash_ai.recognition.resident_models import Pool
    moves=[]
    class Model:
        def resident_modules(self):return [self]
        def parameters(self):return [self]
        def buffers(self):return []
        def numel(self):return 10
        def element_size(self):return 4
        def to(self,device):moves.append(device)
        def close(self):moves.append('close')
    p=Pool(40);a,b=Model(),Model()
    p.put('a',a,'cuda');p.put('b',b,'cuda')
    assert p.used==40 and moves==['cpu','close','cpu']
    assert p.take('b',None,'cuda') is b and p.used==0 and moves[-1]=='cuda'
