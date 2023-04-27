from .SRSolver import SRSolver

def create_solver(mode):
    if mode == 'sr':
        solver = SRSolver()
    else:
        raise NotImplementedError

    return solver
