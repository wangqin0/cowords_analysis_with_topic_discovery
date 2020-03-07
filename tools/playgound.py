class zzy:
    def __init__(self, w, p):
        self.w = w
        self.p = p
        self.s = {'key': 11}


x = zzy(1, 't1')
y = zzy(2, 't2')

xy_dict = {'x': x, 'y': y}

x.s['key'] = 22

y = zzy(3, 't3')
print()