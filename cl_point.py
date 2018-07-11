class CL2DPoint:
    def __init__(self, x=None, y=None, th=None, rho=None):
        self.X = x if x is not None else []
        self.Y = y if y is not None else []
        self.Th = th if th is not None else []
        self.Rho = rho if rho is not None else []


class CL3DPoint(CL2DPoint):
    def __init__(self, x=None, y=None, z=None, th=None, rho=None, ga=None):
        self.Z = z if z is not None else []
        self.Ga = ga if ga is not None else []
        CL2DPoint.__init__(self, x if x is not None else [], y if y is not None else [], th if th is not None else [],
                           rho if rho is not None else [])


class CL2DPointStamped(CL2DPoint):
    def __init__(self):
        self.T = []


class CL3DPointStamped(CL3DPoint):
    def __init__(self):
        self.T = []

