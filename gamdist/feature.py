# Passing untrusted user input may have unintended consequences. 
# Not designed to consume input from unknown sources (i.e., 
# the public internet).

class _Feature:
    def __init__(self, name):
        self._name = name

    def _plot(self):
        pass

    def __str__(self):
        pass

