class TransformerSplitConfig:
    """
    Manages configuration for splitting a transformer model across multiple hosts.

    Methods:
    --------
    __init__(num_splits, hosts, ports, pipe_config):
        Initializes with the number of splits, hosts, ports, and pipeline config.
        Raises ValueError if hosts or ports don't match num_splits.

    get_config():
        Returns the current configuration.

    add_split(host, port):
        Adds a host and port, increasing the number of splits.

    remove_split(index):
        Removes a split by index. Raises IndexError if out of bounds.

    __str__():
        Returns the configuration as a string.
    """
    
    def __init__(self, num_splits, hosts, ports, pipe_config):
        if len(hosts) != num_splits or len(ports) != num_splits:
            raise ValueError("Number of hosts and ports must match the number of splits")
        
        self.config = {
            "num_splits": num_splits,
            "hosts": hosts,
            "ports": ports,
            "pipe_config": pipe_config
            
        }

    def get_config(self):
        return self.config

    def add_split(self, host, port):
        self.config["num_splits"] += 1
        self.config["hosts"].append(host)
        self.config["ports"].append(port)

    def remove_split(self, index):
        if 0 <= index < self.config["num_splits"]:
            self.config["num_splits"] -= 1
            self.config["hosts"].pop(index)
            self.config["ports"].pop(index)
        else:
            raise IndexError("Invalid index")

    def __str__(self):
        return str(self.config)