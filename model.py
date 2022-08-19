import graphs
import model

class Model:
    def __init__(self, *, reps:set=set(), acts:set=set()) -> None:
        self.reps = reps.copy()
        self.acts = acts.copy()
        self.repG = graphs.RepGraph(reps=self.reps)
        self.actG = graphs.ActGraph(acts=self.acts)
        self.saG = graphs.SAGaph(reps=self.reps, acts=self.reps)
    
    def interact(self, reps:set=set()):
        # 直接作为一个state match么？还是去匹配全部相关rep再处理？
        # match - repG
        if reps in self.reps:
            # 1.fully matched
            print(str(reps)+' matched')

        # 2.partial matched
        # if matched - saG
        # action - actG
        # prime? - repG
        # else new
        pass
    
    def abstract(self):
        # merge pathes & simplify reps
        pass