from typing import Dict, Tuple, Callable, List

from ops import NANDOp, InputOp
from scheme import Scheme

def initialize_streams(scheme: Scheme) -> Tuple[Dict[str, InputOp | NANDOp], Dict[str, InputOp]]:
    """
    stream templates of inputs
    Arguments:
        scheme: Scheme = instruction scheme
    Returns:
        ({str: InputOp}, {str: InputOp})
    """
    streams = {i: InputOp(name=i) for i in list(scheme.instruction_set.replace(".", ""))}
    init_val = False
    for i in streams.values():
        i.set_val(init_val)
        init_val = not init_val
    return streams, {k: v for k, v in streams.items()}

def translate_instructions(instructions: str, streams: Dict[str, InputOp | NANDOp]) -> Dict[str, InputOp| NANDOp]:
    """
    streams' final states traced by instructions
    Arguments:
        instructions: str = instruction string
        streams: {str: InputOp}
    Returns:
        {str: InputOp or NANDOp}
    """
    counter = 1
    try:
        while instructions[counter - 1] == ".":
            counter += 1
        while counter < len(instructions):
            if instructions[counter] == ".":
                while instructions[counter] == ".":
                    counter += 1
                counter += 1
            else:
                stream = instructions[counter]
                streams[stream] = NANDOp(streams[instructions[counter - 1]], streams[stream])
                counter += 1
    except IndexError:
        pass
    return streams

def create_func(stream_inputs: Dict[str, InputOp], stream_outputs: Dict[str, InputOp| NANDOp], input_streams: str, output_streams: str) -> Callable[[Tuple[bool]], List[bool]]:
    """
    function created from streams
    Arguments:
        stream_inputs: {str: InputOp} = streams' inputs
        stream_outputs: {str: InputOp or NANDOp} = streams' final states
        input_streams: str = all streams with specified input
        output_streams: str = all streams with monitored output
    Returns:
        ((bool)) -> [bool]
    """
    def func(inputs):
        for i in output_streams:
            stream_outputs[i].reset_state()
        for s, i in zip(input_streams, inputs):
            stream_inputs[s].set_val(i)
        return [stream_outputs[i]() for i in output_streams]
    return func

def instructions_to_function(scheme: Scheme, instructions: str) -> Callable[[Tuple[bool]], List[bool]]:
    """
    function implementing instructions with given scheme
    Arguments:
        scheme: Scheme = instructions scheme
        instructions: str = instructions string
    Returns:
        ((bool)) -> [bool]
    """
    stream_outputs, stream_inputs = initialize_streams(scheme)
    stream_outputs = translate_instructions(instructions, stream_outputs)
    func = create_func(stream_inputs, stream_outputs, scheme.input_streams, scheme.output_streams)
    return func