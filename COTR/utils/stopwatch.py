import time

class StopWatch:
    dic_sum = {}
    dic_count = {}
    stack = []

    @classmethod
    def dump_sum(cls):
        mlen = max( len(k) for k,v in __class__.dic_sum.items() )
        for k, v in __class__.dic_sum.items():
            print(f"[{k:{mlen}}] time: sum:{v:.5f} count:{__class__.dic_count[k]} avg:{v/__class__.dic_count[k]}")

    def __init__(self, name=None):
        if name is None:
            name = f"{self}"
        self.name = name

        if len(__class__.stack)==0:
            self.name_path = f"{self.name}"
        else:
            name_path = '/'.join([st.name for st in __class__.stack])
            self.name_path = name_path + f"/{self.name}"

        if self.name_path not in StopWatch.dic_sum.keys():
            __class__.dic_sum[self.name_path] = 0
            __class__.dic_count[self.name_path] = 0
        self.children = []

    def __enter__(self):
        self.t0 = time.time()
        if len(__class__.stack) > 0:
            __class__.stack[-1].children.append(self)
        __class__.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb ):
        self.t1 = time.time()
        # record
        __class__.dic_sum[self.name_path] += (self.t1 - self.t0)
        __class__.dic_count[self.name_path] += 1
        # print(f"[{self.name}] time: {self.t1-self.t0:.5f}[sec.] sum:{__class__.dic_sum[self.name]:.5f}")
        __class__.stack.pop()
        if len(__class__.stack)==0:
            self.print('')
        #
        return True

    def print(self, name_path):
        print(f"SW:[{'.'*len(name_path)}{self.name}] time: {self.t1-self.t0:.5f}[sec.] sum:{__class__.dic_sum[self.name_path]:.5f}")
        for child in self.children:
            child.print(f"{'.'*len(name_path)}{self.name}")

