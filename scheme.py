class Scheme(object):
    """
    Instruction Scheme stores information necessary to convert instructions to functions
    """
    def __init__(self, instruction_set: str, input_streams: str, output_streams: str) -> None:
        """
        initialize Instruction Scheme
        Arguments:
            instructions_set: str = all possible instructions
            input_streams: str = all streams mapped to an input
            output_streams: str = all streams from which output is read
        Returns:
            None
        """
        self.instruction_set: str = instruction_set
        self.input_streams: str = input_streams
        self.output_streams: str = output_streams

def define_scheme(num_inputs: int, num_outputs: int, extra_streams: int = 2) -> Scheme:
    """
    scheme with num_inputs number of inputs and num_outputs number of outputs
    Arguments:
        num_inputs: int = number of input streams
        num_ouptuts: int = number of output streams
        extra_streams: int = number of extra streams to include in choices
    Returns:
        Scheme
    """
    choices = "abcdefghijklmnopqrstuvwxyz"
    choices = choices + choices.upper()
    inputs = choices[:num_inputs]
    outputs = choices[:num_outputs]
    return Scheme(choices[:max(num_inputs, num_outputs) + extra_streams] + ".", inputs, outputs)