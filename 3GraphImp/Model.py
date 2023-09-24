from RepGraph import RepGraph
from ActGraph import ActGraph
from SAGraph import SAGraph
import utils

class Model:
    
    # sensors = rus
    def __init__(self, sensors:set[str], actions:set[str]) -> None:
        self.reps = RepGraph(sensors)
        self.acts = ActGraph(actions)
        self.sag = SAGraph()

    # activate states
    # locate states in SAG
    # 
    def next(self, ext):
        input = self.rep_label2id(ext)
        self.rep_activate(input)
        self.rep_tick(self, 'conserv')
    
    def rep_label2id(self, input:set[str]):
        return self.reps.to_ids(input)
    
    # when new stimuli comes as set of id
    # with sensation, the input should be all rep units
    def rep_activate(self, input:set[str]):
        self.reps.activate_multi(input, value=1)
    
    # can DIY the flow here
    def rep_tick(self, spread_type):
        if spread_type ==  'conserv':
            self.reps.conserv_spread()
        else:
            self.reps.regular_spread()
        self.reps.activate_func(utils.sigmoid)
        self.reps.deactivate_all(0.2)
    
    def act_exploit(self):
        # match through SAG
        pass

    def act_explore(self):
        pass

    def act_execute(self):
        pass
