class Memory:
    def __init__(self):
        self.data = {}
     
    def read(self, key):
        return self.data.get(key, "")
     
    def append(self, key, value):
        if key in self.data:
            self.data[key] += f"\n{value}"
        else:
            self.data[key] = value
     
    def clear(self):
        self.data = {}