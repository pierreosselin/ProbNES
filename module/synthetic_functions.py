from botorch.test_functions.synthetic import Ackley

def generate_objective(config):
    if config["objective"]["name"] == "ackley":
        return Ackley(config["objective"]["dim"])