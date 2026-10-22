# One-off control (deleted after use): run train_v1 with the pre-fix main() (no exception wrapper).
import os
import sys

sys.path.insert(0, os.getcwd())
import train_v1 as T  # noqa: E402
from absl import app  # noqa: E402


def old_main(_):
    trainer = T.Trainer()
    trainer.setup()
    trainer.run()


app.run(old_main)
