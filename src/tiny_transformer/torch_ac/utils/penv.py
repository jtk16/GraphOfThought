import multiprocessing
from multiprocessing import Process, Pipe
from gymnasium.spaces import Box, Discrete

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs, info = env.reset()
            conn.send((obs, info))
        elif cmd == "close":
            conn.close()
            break
        elif cmd == "get_spaces":
            conn.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv(multiprocessing.Process):
    """A concurrent environment for RL algorithms."""

    def __init__(self, envs):
        super().__init__()
        self.closed = False
        self.nenvs = len(envs)
        self.waiting = [False for _ in envs]
        self.parent_conns, self.child_conns = zip(*[Pipe() for _ in envs])
        self.procs = [Process(target=worker, args=(child_conn, env))
                      for child_conn, env in zip(self.child_conns, envs)]
        for proc in self.procs:
            proc.start()

        self.parent_conns[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.parent_conns[0].recv()

    def step(self, actions):
        for i, action in enumerate(actions):
            self.parent_conns[i].send(("step", action))
        results = [conn.recv() for conn in self.parent_conns]
        obss, rewards, dones, infos = zip(*results)
        return obss, rewards, dones, infos

    def reset(self):
        for i, conn in enumerate(self.parent_conns):
            conn.send(("reset", None))
        results = [conn.recv() for conn in self.parent_conns]
        obss, infos = zip(*results)
        return obss, infos

    def close(self):
        for conn in self.parent_conns:
            conn.send(("close", None))
        for proc in self.procs:
            proc.join()
        self.closed = True