# use this for graphs
class Rep:
    def __init__(self, *, id:int, label:str=None, base:set):
        self.id = id
        self.base = base
    # def getLabel(self)->str:
    #     return self.label
    # def fetId(self)->int:
    #     return self.id
    # def getBase(self)->set:
    #     return self.base


class Act:
    def __init__(self, *, id:int, label:str=None, base:list):
        self.id = id
        if not label:
            self.label = 'a'+str(id)
        else:
            self.label = label
        self.base = base
    # def getLabel(self)->str:
    #     return self.label
    # def fetId(self)->int:
    #     return self.id
    # def getBase(self)->list:
    #     return self.base
    
