import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("m", ROOT / "main.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from kaggle_environments import make
env = make("orbit_wars", configuration={"seed": 42}, debug=False)
env.run([mod.agent, "random"])
f = env.steps[-1]
r0 = f[0].reward
r1 = f[1].reward
print("smoke: r0=%s r1=%s result=%s" % (r0, r1, "WIN" if (r0 or 0) > (r1 or 0) else "LOSS"))
print("Steps:", len(env.steps))
