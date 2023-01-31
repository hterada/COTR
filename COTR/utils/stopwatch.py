from typing import List, Optional, Dict, Any
import time
from collections import OrderedDict
import pandas as pd

class StopWatch:
    """Python の with 構文にもとづくストップウォッチ。withブロック内の所要時間を計測する。
    """
    dic_sum:Dict[str, int] = dict()
    dic_count:Dict[str, int] = dict()
    dic_child_names:Dict[str, OrderedDict] = dict() # map name_path to set(child name)
    stack:List[object] = []

    def __init__(self, name=None):
        if name is None:
            name = f"{self}"
        self.name = name
        self.name_path:List[str] = []

        if len(__class__.stack)==0:
            # root of name_path
            self.name_path.append(self.name)
        else:
            self.name_path.extend( [st.name for st in __class__.stack] )
            self.name_path.append(self.name)

        if str(self.name_path) not in StopWatch.dic_sum.keys():
            __class__.dic_sum[str(self.name_path)] = 0
            __class__.dic_count[str(self.name_path)] = 0
            __class__.dic_child_names[str(self.name_path)] = OrderedDict()


    def __enter__(self):
        if len(__class__.stack) > 0:
            str_parent_name_path = str(__class__.stack[-1].name_path)
            __class__.dic_child_names[str_parent_name_path][self.name] = True
        __class__.stack.append(self)
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb ):
        self.t1 = time.time()
        # record
        __class__.dic_sum[str(self.name_path)] += (self.t1 - self.t0)
        __class__.dic_count[str(self.name_path)] += 1
        __class__.stack.pop()
        if len(__class__.stack)==0:
            self.print(self.name_path)
        #
        return True

    def print(self, name_path:List[str])->None:
        sum = self.__class__.dic_sum[str(name_path)]*1000
        count = self.__class__.dic_count[str(name_path)]
        level = len(name_path)-1

        # parent sum
        parent_sum = self.parent_sum(name_path)

        print(f"SW:[{'.'*level+name_path[-1]:40}] time sum:{sum:8.3f} count:{count:4} avg:{sum/count:8.3f} {'.'*level}{sum/parent_sum*100.0:6.2f}%")
        # child
        for child_name in self.__class__.dic_child_names[str(name_path)].keys():
            child_name_path = list(name_path) #copy
            child_name_path.append(child_name)
            self.print(child_name_path)

    def to_DataFrame(self)->pd.DataFrame:
        l_out:List[Any] = []
        data = self.make_table(l_out, self.name_path)
        columns = ["name", "sum", "count", "avg", "% of parent"]
        df = pd.DataFrame(data, columns=columns)
        return df


    def make_table(self, l_out:List[Any], name_path):
        sum = self.__class__.dic_sum[str(name_path)]*1000
        count = self.__class__.dic_count[str(name_path)]
        level = len(name_path)-1
        root_sum = self.__class__.dic_sum[str(name_path[:1])]*1000
        # add row
        l_out.append(['.'*level+name_path[-1], sum, count ,sum/count ,sum/root_sum*100.0])
        # child
        for child_name in self.__class__.dic_child_names[str(name_path)].keys():
            child_name_path = list(name_path) #copy
            child_name_path.append(child_name)
            self.make_table(l_out, child_name_path)
        return l_out

    def parent_sum(self, child_name_path:List[str])->int:
        if len(child_name_path) < 2:
            # no parent
            parent_name_path = child_name_path
        else:
            parent_name_path = child_name_path[:-1]
        return self.__class__.dic_sum[str(parent_name_path)]*1000

