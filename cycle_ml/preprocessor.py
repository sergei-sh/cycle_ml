
class ToolRecipeCross:
    def __init__(self):
        self._tr_int = {}
        self.cur_idx = -1 

    def convert_next(self, tool_recipe_pair):
       try:
          return self._tr_int[tool_recipe_pair] 
       except KeyError:
          self.cur_idx += 1
          self._tr_int[tool_recipe_pair] = self.cur_idx
          return self.cur_idx

    def get_int(self, tool_recipe_pair):
        return self._tr_int[tool_recipe_pair] 

    @property
    def width(self):
        return self.cur_idx + 1

    def __str__(self):
        out_lst = [(v, k) for k, v in self._tr_int.items()]
        out_lst.sort()
        return str(out_lst)


def get_data(data):
    data.tr_cross = ToolRecipeCross()
    for i in range(0, len(data.tool_recipe)):
        data.tool_recipe[i] = data.tr_cross.convert_next(data.tool_recipe[i])
    return data
